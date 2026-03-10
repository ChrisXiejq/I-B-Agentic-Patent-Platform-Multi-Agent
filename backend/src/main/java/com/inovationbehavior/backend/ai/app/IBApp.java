package com.inovationbehavior.backend.ai.app;

import com.inovationbehavior.backend.ai.advisor.BannedWordsAdvisor;
import com.inovationbehavior.backend.ai.advisor.MyLoggerAdvisor;
import com.inovationbehavior.backend.ai.rag.QueryRewriter;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.Resource;
import org.springframework.beans.factory.annotation.Qualifier;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;

import java.util.List;

@Component
@Slf4j
public class IBApp {

    private final ChatModel chatModel;
    private ChatClient chatClient;

    @Resource
    private BannedWordsAdvisor bannedWordsAdvisor;

    private static final String SYSTEM_PROMPT = """
            You are an intelligent assistant for the patent commercialization platform. You focus on helping users with patent retrieval, value assessment, and commercialization.
            At the opening, briefly introduce yourself and explain you can: query patent details and heat by patent number, help analyze patent technical points and applicable scenarios, and provide personalized advice based on user identity (invitation code/survey).
            During the conversation, you may proactively ask: patent numbers or technical fields of interest, commercialization intentions (license/transfer/equity investment, etc.), target industry or partner type.
            Provide concise, professional advice based on patent information from platform tools, and timely suggest users use the "query patent details" and "query patent heat" capabilities.
            """;

    public IBApp(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    /** 默认注入违禁词等 Advisor 后初始化 ChatClient */
    @PostConstruct
    void initChatClient() {
        MessageWindowChatMemory chatMemory = MessageWindowChatMemory.builder()
                .chatMemoryRepository(new InMemoryChatMemoryRepository())
                .maxMessages(20)
                .build();
        var advisors = new java.util.ArrayList<Advisor>();
        if (bannedWordsAdvisor != null) {
            advisors.add(bannedWordsAdvisor);
        }
        advisors.add(MessageChatMemoryAdvisor.builder(chatMemory).build());
        advisors.add(new MyLoggerAdvisor());
        this.chatClient = ChatClient.builder(chatModel)
                .defaultSystem(SYSTEM_PROMPT)
                .defaultAdvisors(advisors)
                .build();
    }

    /**
     * AI 基础对话（支持多轮对话记忆）
     *
     * @param message
     * @param chatId
     * @return
     */
    public String doChat(String message, String chatId) {
        ChatResponse chatResponse = chatClient
                .prompt()
                .user(message)
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId))
                .call()
                .chatResponse();
        String content = chatResponse.getResult().getOutput().getText();
        log.info("content: {}", content);
        return content;
    }

    /**
     * AI 基础对话（支持多轮对话记忆，SSE 流式传输）
     *
     * @param message
     * @param chatId
     * @return
     */
    public Flux<String> doChatByStream(String message, String chatId) {
        return chatClient
                .prompt()
                .user(message)
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId))
                .stream()
                .content();
    }

    /** 专利咨询报告（结构化输出：标题 + 建议列表） */
    record PatentReport(String title, List<String> suggestions) {
    }

    /**
     * 专利咨询报告（实战结构化输出）
     */
    public PatentReport doChatWithReport(String message, String chatId) {
        PatentReport report = chatClient
                .prompt()
                .system(SYSTEM_PROMPT + "After each conversation, generate a brief patent advisory report: title as 'Patent Advisory Report', content as a list of 3-5 suggestions addressing the user's question, related to patent commercialization, value assessment, or connection advice.")
                .user(message)
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId))
                .call()
                .entity(PatentReport.class);
        log.info("patentReport: {}", report);
        return report;
    }

    // RAG 知识库问答（专利/平台文档）

    @Resource
    private Advisor hybridRagAdvisor;

    @Resource
    @Qualifier("IBVectorStore")
    private VectorStore vectorStore;

    @Resource
    private QueryRewriter queryRewriter;

    /**
     * 和 RAG 知识库进行对话
     *
     * @param message
     * @param chatId
     * @return
     */
    public String doChatWithRag(String message, String chatId) {
        // 查询重写
//        String rewrittenMessage = queryRewriter.doQueryRewrite(message);
        ChatResponse chatResponse = chatClient
                .prompt()
                // 使用改写后的查询
                .user(message)
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId))
                // 开启日志，便于观察效果
                .advisors(new MyLoggerAdvisor())
                // 应用多路召回 RAG：向量检索 + BM25 关键词检索 → RRF 融合 → Rerank
                .advisors(hybridRagAdvisor)
                .call()
                .chatResponse();
        String content = chatResponse.getResult().getOutput().getText();
        log.info("content: {}", content);
        return content;
    }

    // AI 调用工具能力
    @Resource
    private ToolCallback[] allTools;

    /**
     * 专利平台对话（支持调用工具：查专利详情、热度、用户身份等）
     */
    public Flux<String> doChatWithTools(String message, String chatId) {
        return chatClient
                .prompt()
                .user(message)
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId))
                // 开启日志，便于观察效果
                .advisors(new MyLoggerAdvisor())
                .toolCallbacks(allTools)
                .stream()
                .content();
    }

    // AI 调用 MCP 服务

    @Resource
    private ToolCallbackProvider toolCallbackProvider;

    /**
     * 专利平台对话（调用 MCP 服务）
     */
    public String doChatWithMcp(String message, String chatId) {
        ChatResponse chatResponse = chatClient
                .prompt()
                .user(message)
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId))
                // 开启日志，便于观察效果
                .advisors(new MyLoggerAdvisor())
                .toolCallbacks(toolCallbackProvider)
                .call()
                .chatResponse();
        String content = chatResponse.getResult().getOutput().getText();
        log.info("content: {}", content);
        return content;
    }

    // ========== 全能力 Agent 入口：多轮记忆 + RAG 检索增强 + 工具调用 ==========

    /**
     * 全能力 Agent 同步调用：多轮记忆 + RAG（向量+BM25 融合）+ 工具调用（专利详情/热度/用户身份等）
     */
    public String doChatWithFullAgent(String message, String chatId) {
        String rewrittenMessage = queryRewriter != null ? queryRewriter.doQueryRewrite(message) : message;
        ChatResponse chatResponse = chatClient
                .prompt()
                .user(rewrittenMessage)
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId))
                .advisors(new MyLoggerAdvisor())
                .advisors(hybridRagAdvisor)
                .toolCallbacks(allTools)
                .call()
                .chatResponse();
        String content = chatResponse.getResult().getOutput().getText();
        log.info("full-agent sync content length: {}", content != null ? content.length() : 0);
        return content;
    }

    /**
     * 全能力 Agent 流式调用：多轮记忆 + RAG + 工具调用，SSE 流式返回
     */
    public Flux<String> doChatWithFullAgentStream(String message, String chatId) {
        String rewrittenMessage = queryRewriter != null ? queryRewriter.doQueryRewrite(message) : message;
        return chatClient
                .prompt()
                .user(rewrittenMessage)
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId))
                .advisors(new MyLoggerAdvisor())
                .advisors(hybridRagAdvisor)
                .toolCallbacks(allTools)
                .stream()
                .content();
    }
}
