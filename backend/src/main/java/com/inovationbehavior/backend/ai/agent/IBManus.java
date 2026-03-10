package com.inovationbehavior.backend.ai.agent;

import com.inovationbehavior.backend.ai.advisor.MyLoggerAdvisor;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.stereotype.Component;

/**
 * 专利成果转化平台 AI 智能体（支持查专利详情、热度、用户身份等工具，自主规划多步任务）
 */
@Component
public class IBManus extends ToolCallAgent {

    public IBManus(ToolCallback[] allTools, ChatModel genAIChatModel) {
        super(allTools);
        this.setName("patentAssistant");
        String SYSTEM_PROMPT = """
                You are an intelligent assistant for the patent commercialization platform. You are skilled at invoking tools to complete tasks based on user needs.
                Available tools include: get patent details (getPatentDetails), get patent heat/attention (getPatentHeat), get user/survey identity (getUserIdentity), etc.
                Reply in English with concise and professional answers related to patent retrieval, value assessment, and commercialization.
                """;
        this.setSystemPrompt(SYSTEM_PROMPT);
        String NEXT_STEP_PROMPT = """
                Based on the user's question, prioritize selecting appropriate tools (e.g., query patent details first then analyze, or query user identity first then give advice).
                For multi-step tasks, call tools to obtain information first, then give conclusions and next-step suggestions based on the results.
                Call the terminate tool to end the conversation when done.
                """;
        this.setNextStepPrompt(NEXT_STEP_PROMPT);
        this.setMaxSteps(20);
        // 初始化 AI 对话客户端
        ChatClient chatClient = ChatClient.builder(genAIChatModel)
                .defaultAdvisors(new MyLoggerAdvisor())
                .build();
        this.setChatClient(chatClient);
    }
}
