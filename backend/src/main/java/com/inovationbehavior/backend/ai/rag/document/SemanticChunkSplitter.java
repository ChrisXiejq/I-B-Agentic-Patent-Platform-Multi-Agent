package com.inovationbehavior.backend.ai.rag.document;

import org.springframework.ai.document.Document;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * 语义切分 + Token 回退策略：
 * <ul>
 *   <li><b>语义切分</b>：按段落 → 句子边界切分，避免在句中对半截断，保证每块是完整语义单元。</li>
 *   <li><b>Token 回退</b>：当某语义单元超过 maxTokens 时，从末尾回退到上一句/上一段边界，或从末尾收缩至允许长度，避免硬截断导致半句话进入 embedding。</li>
 * </ul>
 * 与纯按字符数递归切分相比，长文信息保留率更高、检索块更连贯。
 */
public class SemanticChunkSplitter implements ChunkSplitter {

    /** 段落分隔：双换行或更多 */
    private static final Pattern PARAGRAPH_SPLIT = Pattern.compile("\\n\\s*\\n");
    /** 句子结束符（中英文） */
    private static final Pattern SENTENCE_END = Pattern.compile("(?<=[。！？.!?\\n])\\s*");

    private final int maxChunkTokens;
    private final int overlapTokens;
    private final TokenCounter tokenCounter;

    public SemanticChunkSplitter(int maxChunkTokens, int overlapTokens, TokenCounter tokenCounter) {
        this.maxChunkTokens = Math.max(100, maxChunkTokens);
        this.overlapTokens = Math.max(0, Math.min(overlapTokens, maxChunkTokens / 3));
        this.tokenCounter = tokenCounter != null ? tokenCounter : new SimpleTokenCounter();
    }

    @Override
    public List<Document> apply(List<Document> documents) {
        if (documents == null || documents.isEmpty()) {
            return List.of();
        }
        List<Document> result = new ArrayList<>();
        for (Document doc : documents) {
            result.addAll(splitOne(doc));
        }
        return result;
    }

    private List<Document> splitOne(Document doc) {
        String text = doc.getText();
        Map<String, Object> meta = doc.getMetadata() != null ? doc.getMetadata() : Map.of();
        if (text == null || text.isBlank()) {
            return List.of();
        }

        List<String> paragraphs = splitKeepDelimiter(PARAGRAPH_SPLIT, text);
        List<String> segments = new ArrayList<>();
        for (String p : paragraphs) {
            p = p.trim();
            if (p.isEmpty()) continue;
            if (tokenCounter.countTokens(p) <= maxChunkTokens) {
                segments.add(p);
            } else {
                List<String> sentences = splitKeepDelimiter(SENTENCE_END, p);
                for (String s : sentences) {
                    s = s.trim();
                    if (!s.isEmpty()) segments.add(s);
                }
            }
        }

        List<Document> chunks = new ArrayList<>();
        List<String> current = new ArrayList<>();
        int currentTokens = 0;

        for (String seg : segments) {
            int segTokens = tokenCounter.countTokens(seg);
            if (segTokens <= maxChunkTokens) {
                if (currentTokens + segTokens > maxChunkTokens && !current.isEmpty()) {
                    flushChunk(current, meta, chunks);
                    current = new ArrayList<>();
                    currentTokens = 0;
                    if (overlapTokens > 0 && !chunks.isEmpty()) {
                        String lastContent = chunks.get(chunks.size() - 1).getText();
                        String overlap = tailByTokens(lastContent, overlapTokens);
                        if (!overlap.isBlank()) {
                            current.add(overlap);
                            currentTokens = tokenCounter.countTokens(overlap);
                        }
                    }
                }
                current.add(seg);
                currentTokens += segTokens;
            } else {
                flushChunk(current, meta, chunks);
                current = new ArrayList<>();
                currentTokens = 0;
                List<String> fallbackChunks = tokenFallback(seg, maxChunkTokens, overlapTokens);
                for (String fc : fallbackChunks) {
                    if (currentTokens + tokenCounter.countTokens(fc) > maxChunkTokens && !current.isEmpty()) {
                        flushChunk(current, meta, chunks);
                        current = new ArrayList<>();
                        currentTokens = 0;
                    }
                    current.add(fc);
                    currentTokens += tokenCounter.countTokens(fc);
                }
            }
        }
        if (!current.isEmpty()) {
            flushChunk(current, meta, chunks);
        }
        return chunks;
    }

    private List<String> tokenFallback(String longText, int maxTokens, int overlap) {
        List<String> out = new ArrayList<>();
        String remaining = longText.trim();
        while (!remaining.isEmpty() && tokenCounter.countTokens(remaining) > maxTokens) {
            int target = maxTokens - overlap;
            String head = headByTokens(remaining, target);
            int cut = findLastSentenceEnd(head);
            if (cut > 0) {
                head = head.substring(0, cut).trim();
            }
            if (!head.isEmpty()) {
                out.add(head);
            }
            int drop = head.length();
            if (drop >= remaining.length()) break;
            remaining = remaining.substring(drop).trim();
        }
        if (!remaining.isEmpty()) {
            out.add(remaining);
        }
        return out;
    }

    private int findLastSentenceEnd(String text) {
        for (int i = text.length() - 1; i >= 0; i--) {
            char c = text.charAt(i);
            if (c == '。' || c == '！' || c == '？' || c == '.' || c == '!' || c == '?' || c == '\n') {
                return i + 1;
            }
        }
        return -1;
    }

    private String headByTokens(String text, int maxTokens) {
        if (tokenCounter.countTokens(text) <= maxTokens) return text;
        int approx = tokensToApproxChars(maxTokens);
        if (approx >= text.length()) return text;
        return text.substring(0, approx);
    }

    private String tailByTokens(String text, int maxTokens) {
        if (tokenCounter.countTokens(text) <= maxTokens) return text;
        int approx = tokensToApproxChars(maxTokens);
        if (approx >= text.length()) return text;
        return text.substring(text.length() - approx);
    }

    private int tokensToApproxChars(int tokens) {
        return tokens * 2;
    }

    private void flushChunk(List<String> parts, Map<String, Object> meta, List<Document> out) {
        if (parts.isEmpty()) return;
        String content = String.join("\n\n", parts);
        Map<String, Object> chunkMeta = new java.util.HashMap<>(meta);
        out.add(new Document(content, chunkMeta));
    }

    private static List<String> splitKeepDelimiter(Pattern pattern, String text) {
        List<String> list = new ArrayList<>();
        String[] parts = pattern.split(text);
        for (String p : parts) {
            String t = p.trim();
            if (!t.isEmpty()) list.add(t);
        }
        return list;
    }
}
