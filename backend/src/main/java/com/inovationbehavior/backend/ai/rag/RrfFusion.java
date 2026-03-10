package com.inovationbehavior.backend.ai.rag;

import org.springframework.ai.document.Document;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Reciprocal Rank Fusion (RRF) 融合多路召回结果
 * RRF 公式: score(d) = sum_i 1 / (k + rank_i(d))，通常 k=60
 */
public final class RrfFusion {

    public static final int DEFAULT_K = 60;

    /**
     * 对多路召回的文档列表按 RRF 分数融合并去重
     *
     * @param rankedLists 多路召回的文档列表（每路已按相关性排序）
     * @param k           RRF 常数，默认 60
     * @return 融合后按 RRF 分数降序排列的文档列表（去重，同一 Document 只保留一份）
     */
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

    /**
     * 使用默认 k=60 进行融合
     */
    public static List<Document> fuse(List<List<Document>> rankedLists) {
        return fuse(rankedLists, DEFAULT_K);
    }

    private record RrfEntry(Document doc, double score) {}
}
