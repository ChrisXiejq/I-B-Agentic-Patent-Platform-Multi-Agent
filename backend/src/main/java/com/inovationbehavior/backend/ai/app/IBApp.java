package com.inovationbehavior.backend.ai.app;

import com.inovationbehavior.backend.ai.advisor.BannedWordsAdvisor;
import com.inovationbehavior.backend.ai.advisor.MyLoggerAdvisor;
import com.inovationbehavior.backend.ai.agent.GraphTaskAgent;
import com.inovationbehavior.backend.ai.rag.preretrieval.QueryRewriter;
import com.inovationbehavior.backend.ai.app.ReplanService;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Component;
import com.inovationbehavior.backend.ai.graph.PatentGraphRunner;

import java.util.List;

@Component
@Slf4j
public class IBApp {

    private final ChatModel chatModel;

    @Autowired(required = false)
    private PatentGraphRunner patentGraphRunner;
    private ChatClient chatClient;

    @Resource
    private BannedWordsAdvisor bannedWordsAdvisor;

    private static final String SYSTEM_PROMPT = """
            You are an intelligent assistant for the patent commercialization platform. You focus on helping users with patent retrieval, value assessment, and commercialization.
            At the opening, briefly introduce yourself and explain you can: query patent details and heat by patent number, help analyze patent technical points and applicable scenarios, and provide personalized advice based on user identity (invitation code/survey).
            During the conversation, you may proactively ask: patent numbers or technical fields of interest, commercialization intentions (license/transfer/equity investment, etc.), target industry or partner type.
            Provide concise, professional advice based on patent information from platform tools, and timely suggest users use the "query patent details" and "query patent heat" capabilities.
            When you need to recall what the user said earlier or this conversation's history, call the retrieve_history tool with the current conversation_id and a short query (e.g. patent number or topic).
            """;

    public IBApp(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    /** 默认注入违禁词等 Advisor 后初始化 ChatClient；对话上下文统一由三层记忆 + retrieve_history 提供，不再使用 MessageWindowChatMemory */
    @PostConstruct
    void initChatClient() {
        var advisors = new java.util.ArrayList<Advisor>();
        if (bannedWordsAdvisor != null) {
            advisors.add(bannedWordsAdvisor);
        }
        advisors.add(new MyLoggerAdvisor());
        this.chatClient = ChatClient.builder(chatModel)
                .defaultSystem(SYSTEM_PROMPT)
                .defaultAdvisors(advisors)
                .build();
    }

    @Resource
    private Advisor hybridRagAdvisor;

    @Resource
    @Qualifier("retrievalExpertRagAdvisor")
    private Advisor retrievalExpertRagAdvisor;

    @Resource
    private QueryRewriter queryRewriter;

    // AI 调用工具能力
    @Resource
    private ToolCallback[] allTools;

    @Autowired(required = false)
    @Qualifier("memoryPersistenceAdvisor")
    private Advisor memoryPersistenceAdvisor;

    @Autowired(required = false)
    @Qualifier("agentTraceAdvisor")
    private Advisor agentTraceAdvisor;

    @Resource
    private ReplanService replanService;

    /**
     * 多 Agent 图编排入口，提供给controller的入口
     */
    public String doChatWithMultiAgentOrFull(String message, String chatId) {
        return patentGraphRunner.run(message, chatId);
    }

    private static final String RETRIEVAL_EXPERT_PROMPT = """
            You are the retrieval expert of the patent platform. Your job is to fetch patent details, patent heat, knowledge base content, and when needed, up-to-date or general knowledge from the web.
            - Use getPatentDetails, getPatentHeat, and retrieve_history for patent-specific or conversation history.
            - Use searchWeb (web search) when: the user asks for general/conceptual info (e.g. what is patent commercialization, platform introduction), or when RAG context is missing or insufficient. Prefer searchWeb for broad or latest-information questions.
            Reply briefly with the retrieved data; do not give commercialization advice.
            """;
    private static final String ANALYSIS_EXPERT_PROMPT = """
            You are the technical/value analysis expert. Based on patent details and heat already available or that you fetch, analyze technical points, applicability, and value.
            Use getPatentDetails, getPatentHeat, retrieve_history, and RAG when needed. Reply with concise analysis; do not give licensing or partnership advice.
            """;
    private static final String ADVICE_EXPERT_PROMPT = """
            You are the commercialization advisor. Give licensing, transfer, or partnership advice based on user identity and patent context.
            Use getUserIdentity, getPatentDetails, getPatentHeat when needed. Reply with actionable advice.
            """;

    /** 图内专家 Agent 的下一步提示（think/act 循环中）：完成本任务、必要时调用工具、完成后可调用 doTerminate。 */
    private static final String GRAPH_TASK_NEXT_STEP_PROMPT = """
            Complete the current task using the available tools as needed. When you have enough information, summarize briefly for the user.
            Call doTerminate when the task is done.
            """;

    /**
     * 按任务类型返回对应 system prompt（Executor 单节点按 task 选 prompt）。
     */
    public String getPromptForTask(String task) {
        if (task == null) return getRetrievalExpertPrompt();
        String t = task.trim().toLowerCase();
        if (t.contains("retrieval")) return getRetrievalExpertPrompt();
        if (t.contains("analysis")) return getAnalysisExpertPrompt();
        if (t.contains("advice")) return getAdviceExpertPrompt();
        return getRetrievalExpertPrompt();
    }

    /**
     * Executor 单任务 ReAct 执行：使用 IBManus 架构（GraphTaskAgent think→act 多步循环），
     * 按任务类型注入 RAG/记忆/Trace，若 retrieval 上一步不足则注入 searchWeb 补足提示。
     */
    public String doReActForTask(String task, String message, String chatId, List<String> stepResults) {
        String systemPrompt = getPromptForTask(task);
        String effectiveMessage = message;
        // 内部接口下线，调用websearch的兜底
        if (task != null && task.toLowerCase().contains("retrieval") && stepResults != null && !stepResults.isEmpty()) {
            String lastResult = stepResults.get(stepResults.size() - 1);
            if (replanService != null && ReplanService.shouldRetryRetrievalWithWeb(lastResult)) {
                effectiveMessage = "[上一轮检索无有效结果（接口失败或无数据），请改用 searchWeb 检索该专利或相关公开信息后简要回复。] 用户问题：" + (message != null ? message : "");
                log.info("[AgentGraph.IBApp] 检索重试：注入 searchWeb 补足提示，原消息长度={}", message != null ? message.length() : 0);
            }
        }
        String rewritten = queryRewriter != null ? queryRewriter.doQueryRewrite(effectiveMessage) : effectiveMessage;
        Advisor ragAdvisor = systemPrompt.contains("retrieval") && retrievalExpertRagAdvisor != null
                ? retrievalExpertRagAdvisor : hybridRagAdvisor;
        GraphTaskAgent agent = new GraphTaskAgent(
                allTools, chatModel, systemPrompt, GRAPH_TASK_NEXT_STEP_PROMPT, chatId,
                ragAdvisor, memoryPersistenceAdvisor, agentTraceAdvisor);
        String out = agent.run(rewritten);
        String expertName = systemPrompt.contains("retrieval") ? "Retrieval" : systemPrompt.contains("analysis") ? "Analysis" : "Advice";
        log.info("[AgentGraph.IBApp][doReActForTask] 专家调用(IBManus) expert={} messageLength={} responseLength={}",
                expertName, effectiveMessage != null ? effectiveMessage.length() : 0, out != null ? out.length() : 0);
        return out != null ? out : "";
    }

    /** 简单问候或“介绍自己”类短句，无需走检索专家，直接 synthesize 即可 */
    private static boolean isSimpleGreetingOrIntro(String userMessage) {
        if (userMessage == null) return false;
        String s = userMessage.trim();
        if (s.length() > 50) return false;
        String lower = s.toLowerCase();
        return lower.contains("你好") || lower.contains("嗨") || lower.contains("hi ") || lower.equals("hi")
                || lower.contains("介绍自己") || lower.contains("自我介绍") || lower.contains("你是谁")
                || lower.contains("introduce yourself");
    }

    private static String abbreviate(String s, int maxLen) {
        if (s == null) return "null";
        s = s.trim();
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen) + "...";
    }

    /**
     * 综合节点：根据已有 stepResults 与用户问题，生成最终回复。
     */
    public String synthesizeAnswer(String userMessage, List<String> stepResults, String chatId) {
        String context = stepResults == null || stepResults.isEmpty()
                ? "No prior expert outputs."
                : String.join("\n---\n", stepResults);
        String prompt = """
                You are the patent platform assistant. Below are expert outputs from retrieval/analysis/advice agents.
                Provide a concise, friendly final answer to the user. Do not repeat long raw data; summarize and give clear next steps if needed.
                User question: %s
                Expert outputs:
                %s
                """.formatted(userMessage, context);
        ChatResponse resp = chatClient.prompt()
                .user(prompt)
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId))
                .call()
                .chatResponse();
        String content = resp.getResult().getOutput().getText();
        log.info("[AgentGraph.IBApp][synthesizeAnswer] 综合节点 stepResultsCount={} finalAnswerLength={}",
                stepResults != null ? stepResults.size() : 0, content != null ? content.length() : 0);
        return content != null ? content : "";
    }

    // ========== P&E：规划与重规划 ==========

    private static final String PLAN_PROMPT = """
            You are the planner for a patent platform agent. Given the user message and optional prior step results, output a comma-separated list of steps to execute. Each step is exactly one of: retrieval, analysis, advice, synthesize.
            - retrieval: fetch patent details/heat, knowledge base, or web search when needed.
            - analysis: technical/value analysis of a patent.
            - advice: commercialization, licensing, or partnership advice.
            - synthesize: generate final answer and end (use when question is simple greeting, thanks, goodbye, or when we already have enough context).
            Rules: Use minimal steps. For simple greeting or "who are you" reply only: synthesize. For "query patent then analyze" reply: retrieval,analysis,synthesize. For full pipeline: retrieval,analysis,advice,synthesize.
            Reply with only the comma-separated list, e.g. retrieval,analysis,synthesize or synthesize.
            User message: %s
            Prior step results (if any): %s
            """;
    private static final String CHECK_ENV_PROMPT = """
            Given the last task result and the user question, does the result indicate an "environment change" that should change our remaining plan?
            Environment change includes: patent is invalid/expired/withdrawn (专利已失效/过期/撤回), no patent data, authorization failed, or similar. Answer with exactly one word: yes or no.
            Last task result: %s
            User message: %s
            Reply: yes or no
            """;

    /**
     * P&E 初始规划：根据用户问题（及已有 stepResults，通常为空）生成执行计划。
     */
    public List<String> createPlan(String userMessage, List<String> stepResults) {
        if (userMessage == null) userMessage = "";
        if (stepResults == null) stepResults = List.of();
        if (isSimpleGreetingOrIntro(userMessage)) {
            log.info("[AgentGraph.IBApp][createPlan] 规划 识别为简单问候，直接 synthesize");
            return List.of("synthesize");
        }
        String prior = stepResults.isEmpty() ? "None" : String.join("\n---\n", stepResults);
        String prompt = PLAN_PROMPT.formatted(userMessage, prior);
        ChatResponse resp = chatClient.prompt().user(prompt).call().chatResponse();
        String raw = resp.getResult().getOutput().getText();
        List<String> plan = parsePlan(raw);
        log.info("[AgentGraph.IBApp][createPlan] 规划 userMessage(preview)= {} -> plan={}", abbreviate(userMessage, 50), plan);
        return plan.isEmpty() ? List.of("synthesize") : plan;
    }

    /**
     * 检查上一步执行结果是否表明「环境变化」（如专利已失效），需动态更新剩余任务。
     */
    public boolean checkEnvironmentChange(String lastStepResult, List<String> remainingTasks, String userMessage) {
        if (lastStepResult == null || lastStepResult.isBlank()) return false;
        String lower = lastStepResult.trim().toLowerCase();
        if (lower.contains("专利已失效") || lower.contains("已过期") || lower.contains("已撤回") || lower.contains("invalid") || lower.contains("expired") || lower.contains("withdrawn")) return true;
        if (lower.contains("无此专利") || lower.contains("未找到专利") || lower.contains("no patent found")) return true;
        if (remainingTasks == null || remainingTasks.isEmpty()) return false;
        String prompt = CHECK_ENV_PROMPT.formatted(abbreviate(lastStepResult, 500), userMessage != null ? abbreviate(userMessage, 200) : "");
        ChatResponse resp = chatClient.prompt().user(prompt).call().chatResponse();
        String raw = resp.getResult().getOutput().getText();
        boolean yes = raw != null && raw.trim().toLowerCase().startsWith("yes");
        log.info("[AgentGraph.IBApp][checkEnvironmentChange] 环境检查 lastResultLen={} remainingSize={} -> environmentChanged={}", lastStepResult.length(), remainingTasks.size(), yes);
        return yes;
    }

    private static List<String> parsePlan(String raw) {
        if (raw == null) return List.of();
        List<String> out = new java.util.ArrayList<>();
        for (String s : raw.trim().toLowerCase().split("[,，\\s]+")) {
            String t = s.trim();
            if (t.isEmpty()) continue;
            if (t.contains("retrieval")) out.add("retrieval");
            else if (t.contains("analysis")) out.add("analysis");
            else if (t.contains("advice")) out.add("advice");
            else if (t.contains("synthesize") || t.contains("end")) out.add("synthesize");
        }
        if (!out.isEmpty() && !"synthesize".equals(out.get(out.size() - 1))) {
            out.add("synthesize");
        }
        return out;
    }

    public String getRetrievalExpertPrompt() { return RETRIEVAL_EXPERT_PROMPT; }
    public String getAnalysisExpertPrompt() { return ANALYSIS_EXPERT_PROMPT; }
    public String getAdviceExpertPrompt() { return ADVICE_EXPERT_PROMPT; }

    /**
     * 和 RAG 知识库进行对话
     * @param message
     * @param chatId
     * @return
     * 仅用来跑RAGAS测试，不对外提供controller
     */
    public String doChatWithRag(String message, String chatId) {
        ChatResponse chatResponse = chatClient
                .prompt()
                .user(message)
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId))
                .advisors(hybridRagAdvisor)
                .call()
                .chatResponse();
        String content = chatResponse.getResult().getOutput().getText();
        log.info("content: {}", content);
        return content;
    }
}
