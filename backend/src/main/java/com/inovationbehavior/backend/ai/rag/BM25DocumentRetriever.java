package com.inovationbehavior.backend.ai.rag;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * 基于 BM25 算法的关键词检索器（面向英文文档）
 * 使用倒排索引优化；建索引时过滤停用词与过短 token，控制词表量级。
 */
public class BM25DocumentRetriever implements DocumentRetriever {

    private static final Pattern TOKEN_PATTERN = Pattern.compile("[\\p{L}]+|[\\p{N}]+");
    private static final double K1 = 1.2;
    private static final double B = 0.75;
    /** 仅索引长度 >= 2 的 token，减少词表 */
    private static final int MIN_TERM_LENGTH = 2;
    /** 英文停用词（小写），不进入倒排索引（无重复） */
    private static final Set<String> STOP_WORDS = Set.of(
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
            "its", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with", "this", "but",
            "they", "have", "had", "been", "being", "do", "does", "did", "would", "could", "should",
            "may", "might", "must", "can", "need", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "every", "both", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "if", "because",
            "until", "while", "about", "over", "which", "who", "whom", "what", "these",
            "those", "them", "their", "we", "our", "you", "your", "she", "her", "his", "him", "i"
    );

    private final List<Document> documents;
    private final int topK;
    private final int totalDocs;
    private final double avgDocLength;
    /** 文档长度数组，docLengths[docId] = 该文档词数 */
    private final int[] docLengths;
    /** 倒排索引：term -> (docId -> tf)，仅包含至少出现一次的 (docId, tf) */
    private final Map<String, Map<Integer, Integer>> invertedIndex;
    /** 词文档频率：term -> 包含该词的文档数 */
    private final Map<String, Integer> docFreq;

    public BM25DocumentRetriever(List<Document> documents, int topK) {
        this.documents = Objects.requireNonNull(documents);
        this.topK = Math.max(1, topK);
        this.totalDocs = documents.size();
        if (documents.isEmpty()) {
            this.docLengths = new int[0];
            this.avgDocLength = 0;
            this.invertedIndex = Map.of();
            this.docFreq = Map.of();
        } else {
            this.docLengths = new int[totalDocs];
            IndexBuildResult result = buildInvertedIndex();
            this.avgDocLength = result.avgDocLength;
            this.invertedIndex = result.invertedIndex;
            this.docFreq = result.docFreq;
        }
    }

    private IndexBuildResult buildInvertedIndex() {
        Map<String, Map<Integer, Integer>> invIndex = new HashMap<>();
        long totalLength = 0;

        for (int docId = 0; docId < documents.size(); docId++) {
            Document doc = documents.get(docId);
            List<String> terms = tokenize(doc.getText());
            int len = terms.size();
            docLengths[docId] = len;
            totalLength += len;

            Map<String, Integer> termFreq = new HashMap<>();
            for (String t : terms) {
                termFreq.merge(t, 1, Integer::sum);
            }
            for (Map.Entry<String, Integer> e : termFreq.entrySet()) {
                if (shouldIndex(e.getKey())) {
                    invIndex.computeIfAbsent(e.getKey(), k -> new HashMap<>()).put(docId, e.getValue());
                }
            }
        }

        double avg = totalDocs > 0 ? (double) totalLength / totalDocs : 0;
        Map<String, Integer> df = new HashMap<>();
        for (String term : invIndex.keySet()) {
            df.put(term, invIndex.get(term).size());
        }
        return new IndexBuildResult(invIndex, df, avg);
    }

    private List<String> tokenize(String text) {
        if (text == null || text.isBlank()) return List.of();
        return TOKEN_PATTERN.matcher(text).results()
                .map(m -> m.group().toLowerCase())
                .filter(t -> t.length() > 0)
                .collect(Collectors.toList());
    }

    /** 仅索引长度足够且非停用词的 token，用于控制词表量级 */
    private static boolean shouldIndex(String term) {
        return term != null && term.length() >= MIN_TERM_LENGTH && !STOP_WORDS.contains(term);
    }

    @Override
    public List<Document> retrieve(Query query) {
        if (documents.isEmpty()) return List.of();

        String queryText = query.text();
        if (queryText == null || queryText.isBlank()) return List.of();

        List<String> qTerms = tokenize(queryText);
        if (qTerms.isEmpty()) return List.of();

        // 去重，且只保留会出现在索引中的词（长度>=2 且非停用词）
        Set<String> uniqueQTerms = qTerms.stream()
                .filter(BM25DocumentRetriever::shouldIndex)
                .collect(Collectors.toCollection(LinkedHashSet::new));
        if (uniqueQTerms.isEmpty()) return List.of();

        // 候选文档 = 至少包含一个查询词的文档（从倒排索引取并集）
        Set<Integer> candidateDocIds = new HashSet<>();
        for (String term : uniqueQTerms) {
            Map<Integer, Integer> postings = invertedIndex.get(term);
            if (postings != null) {
                candidateDocIds.addAll(postings.keySet());
            }
        }

        if (candidateDocIds.isEmpty()) return List.of();

        List<ScoredDoc> scored = new ArrayList<>(candidateDocIds.size());
        for (Integer docId : candidateDocIds) {
            Document doc = documents.get(docId);
            int docLen = docLengths[docId];
            double score = 0;

            for (String term : uniqueQTerms) {
                Map<Integer, Integer> postings = invertedIndex.get(term);
                if (postings == null) continue;
                Integer tf = postings.get(docId);
                if (tf == null || tf == 0) continue;

                int df = docFreq.getOrDefault(term, 0);
                if (df == 0) continue;

                double idf = Math.log(1 + (totalDocs - df + 0.5) / (df + 0.5));
                double numerator = tf * (K1 + 1);
                double denominator = tf + K1 * (1 - B + B * (double) docLen / avgDocLength);
                score += idf * numerator / denominator;
            }

            if (score > 0) {
                scored.add(new ScoredDoc(doc, score));
            }
        }

        return scored.stream()
                .sorted(Comparator.<ScoredDoc>comparingDouble(s -> s.score).reversed())
                .limit(topK)
                .map(s -> s.doc)
                .collect(Collectors.toList());
    }

    private record IndexBuildResult(
            Map<String, Map<Integer, Integer>> invertedIndex,
            Map<String, Integer> docFreq,
            double avgDocLength
    ) {}

    private record ScoredDoc(Document doc, double score) {}
}
