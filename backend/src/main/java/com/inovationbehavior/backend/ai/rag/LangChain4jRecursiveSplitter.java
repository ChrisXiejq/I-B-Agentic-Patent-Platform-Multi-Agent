package com.inovationbehavior.backend.ai.rag;

import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import org.springframework.ai.document.Document;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 使用 LangChain4j 现成的 DocumentSplitters.recursive() 做递归分块（段落→行→句→词→字符 + overlap）。
 * 将 Spring AI 的 Document 转为 LangChain4j Document，分块后再转回 Spring AI Document。
 */
public class LangChain4jRecursiveSplitter {

    private final DocumentSplitter splitter;

    public LangChain4jRecursiveSplitter(int maxSegmentSizeInChars, int maxOverlapSizeInChars) {
        this.splitter = DocumentSplitters.recursive(maxSegmentSizeInChars, maxOverlapSizeInChars);
    }

    /**
     * 对 Spring AI Document 列表分块，返回块级 Spring AI Document 列表。
     */
    public List<Document> apply(List<Document> documents) {
        if (documents == null || documents.isEmpty()) return List.of();

        List<Document> result = new ArrayList<>();
        for (Document springDoc : documents) {
            dev.langchain4j.data.document.Document lc4jDoc = toLangChain4jDocument(springDoc);
            List<TextSegment> segments = splitter.split(lc4jDoc);
            for (TextSegment seg : segments) {
                result.add(toSpringAiDocument(seg));
            }
        }
        return result;
    }

    private static dev.langchain4j.data.document.Document toLangChain4jDocument(Document doc) {
        String text = doc.getText();
        Map<String, Object> meta = doc.getMetadata();
        if (meta == null || meta.isEmpty()) {
            return dev.langchain4j.data.document.Document.from(text);
        }
        Metadata lcMeta = new Metadata();
        meta.forEach((k, v) -> {
            if (v != null) lcMeta.put(k, v.toString());
        });
        return dev.langchain4j.data.document.Document.from(text, lcMeta);
    }

    private static Document toSpringAiDocument(TextSegment seg) {
        Map<String, Object> meta = new java.util.HashMap<>();
        if (seg.metadata() != null && seg.metadata().toMap() != null) {
            seg.metadata().toMap().forEach((k, v) -> meta.put(k, v));
        }
        return new Document(seg.text(), meta);
    }
}
