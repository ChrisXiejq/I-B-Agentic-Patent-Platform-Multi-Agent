package com.inovationbehavior.backend.ai.agent;

import com.inovationbehavior.backend.ai.advisor.MyLoggerAdvisor;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;

import java.util.ArrayList;
import java.util.List;

/**
 * 图内单任务专家 Agent：复用 IBManus 架构（BaseAgent → ReActAgent → ToolCallAgent），
 * 按任务类型配置 systemPrompt，注入 RAG/记忆/Trace 等 Advisor，执行 think→act 多步循环。
 * run() 返回最后一条助手消息文本，供图内 Synthesize 使用。
 */
public class GraphTaskAgent extends ToolCallAgent {

    public GraphTaskAgent(
            ToolCallback[] allTools,
            ChatModel chatModel,
            String systemPrompt,
            String nextStepPrompt,
            String chatId,
            Advisor ragAdvisor,
            Advisor memoryPersistenceAdvisor,
            Advisor agentTraceAdvisor,
            Advisor persistingTraceAdvisor) {
        super(allTools);
        setName("graphTask");
        setSystemPrompt(systemPrompt != null ? systemPrompt : "");
        setNextStepPrompt(nextStepPrompt != null ? nextStepPrompt : "");
        setMaxSteps(5);
        setConversationId(chatId);
        List<Advisor> advisors = new ArrayList<>();
        if (memoryPersistenceAdvisor != null) {
            advisors.add(memoryPersistenceAdvisor);
        }
        if (agentTraceAdvisor != null) {
            advisors.add(agentTraceAdvisor);
        }
        if (persistingTraceAdvisor != null) {
            advisors.add(persistingTraceAdvisor);
        }
        if (ragAdvisor != null) {
            advisors.add(ragAdvisor);
        }
        setExtraAdvisors(advisors);
        ChatClient client = ChatClient.builder(chatModel)
                .defaultAdvisors(new MyLoggerAdvisor())
                .build();
        setChatClient(client);
    }

    /**
     * 执行 think→act 循环，返回最后一条助手消息文本（供图内 Synthesize），若无则返回步骤摘要。
     */
    @Override
    public String run(String userPrompt) {
        String fullLog = super.run(userPrompt);
        List<Message> list = getMessageList();
        for (int i = list.size() - 1; i >= 0; i--) {
            if (list.get(i) instanceof AssistantMessage am) {
                String text = am.getText();
                if (text != null && !text.isBlank()) {
                    return text;
                }
            }
        }
        return fullLog != null ? fullLog : "";
    }
}
