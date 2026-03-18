package com.inovationbehavior.backend.ai.rag.hyde;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;

import java.util.List;

/**
 * HyDE (Hypothetical Document Embeddings)：用 LLM 根据 query 生成「假设性文档」再做向量检索，
 * 使检索与文档空间对齐，提升召回。
 */
@Slf4j
public class HyDEDocumentRetriever implements DocumentRetriever {

    private static final int MAX_HYPOTHETICAL_CHARS = 500;

    private final VectorStoreDocumentRetriever vectorRetriever;
    private final ChatModel chatModel;

    public HyDEDocumentRetriever(VectorStoreDocumentRetriever vectorRetriever, ChatModel chatModel) {
        this.vectorRetriever = vectorRetriever;
        this.chatModel = chatModel;
    }

    @Override
    public List<Document> retrieve(Query query) {
        String queryText = query != null ? query.getQueryText() : "";
        if (queryText == null || queryText.isBlank()) {
            return vectorRetriever.retrieve(query);
        }
        String hypothetical = generateHypotheticalDocument(queryText);
        if (hypothetical == null || hypothetical.isBlank()) {
            return vectorRetriever.retrieve(query);
        }
        Query hydeQuery = new Query(hypothetical);
        return vectorRetriever.retrieve(hydeQuery);
    }

    private String generateHypotheticalDocument(String queryText) {
        String prompt = """
                Write a short paragraph (2-4 sentences) that would be an ideal answer to the following question. \
                Write only the paragraph, no prefix or explanation. Use the same language as the question.
                Question: %s
                """.formatted(truncate(queryText, 400));
        try {
            ChatResponse resp = chatModel.call(new Prompt(prompt));
            String out = resp.getResult() != null && resp.getResult().getOutput() != null
                    ? resp.getResult().getOutput().getText() : "";
            out = out.trim();
            if (out.length() > MAX_HYPOTHETICAL_CHARS) {
                out = out.substring(0, MAX_HYPOTHETICAL_CHARS) + "...";
            }
            return out;
        } catch (Exception e) {
            log.warn("[HyDE] Failed to generate hypothetical document: {}", e.getMessage());
            return null;
        }
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        return s.length() <= max ? s : s.substring(0, max) + "...";
    }
}
