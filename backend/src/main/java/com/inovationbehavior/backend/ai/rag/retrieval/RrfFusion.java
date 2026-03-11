package com.inovationbehavior.backend.ai.rag.retrieval;

import org.springframework.ai.document.Document;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 检索：Reciprocal Rank Fusion (RRF) 融合多路召回结果
 * RRF 公式: score(d) = sum_i 1 / (k + rank_i(d))，通常 k=60
 */
public final class RrfFusion {

    public static final int DEFAULT_K = 60;

    public static List<Document> fuse(List<List<Document>> rankedLists, int k) {
        if (rankedLists == null || rankedLists.isEmpty()) return List.of();

        Map<String, RrfEntry> idToEntry = new LinkedHashMap<>();

        for (List<Document> list : rankedLists) {
            if (list == null) continue;
            for (int rank = 0; rank < list.size(); rank++) {
                Document doc = list.get(rank);
                if (doc == null) continue;
                String id = doc.getId();
                if (id == null || id.isBlank()) {
                    String content = doc.getText();
                    id = System.identityHashCode(doc) + "_" + (content != null ? content.hashCode() : rank);
                }
                double rrf = 1.0 / (k + rank + 1);
                idToEntry.merge(id, new RrfEntry(doc, rrf), (a, b) -> new RrfEntry(a.doc, a.score + b.score));
            }
        }

        return idToEntry.values().stream()
                .sorted(Comparator.<RrfEntry>comparingDouble(e -> e.score).reversed())
                .map(e -> e.doc)
                .collect(Collectors.toList());
    }

    public static List<Document> fuse(List<List<Document>> rankedLists) {
        return fuse(rankedLists, DEFAULT_K);
    }

    private record RrfEntry(Document doc, double score) {}
}
