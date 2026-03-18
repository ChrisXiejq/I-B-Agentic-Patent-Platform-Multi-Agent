package com.inovationbehavior.backend.ai.rag.graphrag;

import org.springframework.ai.document.Document;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 内存实现的实体→文档索引，用于 GraphRAG 局部检索。建索引时对每个文档抽取实体并建立 entity -> docs 映射。
 */
public class InMemoryEntityDocumentIndex implements EntityDocumentIndex {

    /** entity (normalized) -> list of documents containing it */
    private final Map<String, List<Document>> entityToDocs = new ConcurrentHashMap<>();

    public void buildFromDocuments(List<Document> documents, EntityExtractor extractor) {
        if (documents == null || extractor == null) return;
        for (Document doc : documents) {
            String text = doc.getText();
            if (text == null || text.isBlank()) continue;
            List<String> entities = extractor.extractFromText(text);
            for (String entity : entities) {
                String key = normalize(entity);
                if (key.isBlank()) continue;
                entityToDocs.computeIfAbsent(key, k -> new ArrayList<>()).add(doc);
            }
        }
    }

    @Override
    public List<Document> getDocumentsForEntities(List<String> entities, int topK) {
        if (entities == null || entities.isEmpty()) return List.of();
        Map<Document, Integer> docScore = new HashMap<>();
        for (String entity : entities) {
            String key = normalize(entity);
            List<Document> docs = entityToDocs.get(key);
            if (docs != null) {
                for (Document d : docs) {
                    docScore.merge(d, 1, Integer::sum);
                }
            }
        }
        return docScore.entrySet().stream()
                .sorted(Map.Entry.<Document, Integer>comparingByValue().reversed())
                .limit(topK)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }

    private static String normalize(String s) {
        if (s == null) return "";
        return s.trim().toLowerCase();
    }
}
