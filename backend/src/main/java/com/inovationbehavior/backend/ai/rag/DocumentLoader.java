package com.inovationbehavior.backend.ai.rag;

import cn.hutool.crypto.digest.DigestUtil;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.markdown.MarkdownDocumentReader;
import org.springframework.ai.reader.markdown.config.MarkdownDocumentReaderConfig;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * RAG 文档加载器：从 documents 目录加载所有 .md 文档，
 * 使用 LangChain4j DocumentSplitters.recursive() 进行分块（段落→行→句→词→字符 + overlap）。
 * 分块结果供 pgvector 向量库和 BM25 内存倒排索引共用。
 */
@Component
@Slf4j
public class DocumentLoader {

    private final ResourcePatternResolver resourcePatternResolver;
    private final LangChain4jRecursiveSplitter splitter;

    @Value("${app.rag.documents.path:classpath*:documents/**/*.md}")
    private String documentsPath;

    public DocumentLoader(ResourcePatternResolver resourcePatternResolver,
                          LangChain4jRecursiveSplitter splitter) {
        this.resourcePatternResolver = resourcePatternResolver;
        this.splitter = splitter;
    }

    /**
     * 加载 documents 目录下所有 .md 文件，经 LangChain4j 递归分块，返回块级 Document 列表。
     * 该列表将同时写入 pgvector 和用于 BM25 倒排索引。
     */
    public List<Document> loadDocumentsForRag() {
        List<Document> raw = loadMarkdownDocuments();
        if (raw.isEmpty()) {
            log.warn("No Markdown files found at [{}], RAG corpus will be empty.", documentsPath);
            return List.of();
        }
        List<Document> chunks = splitter.apply(raw);
        for (Document chunk : chunks) {
            String source = String.valueOf(chunk.getMetadata().getOrDefault("source", ""));
            String chunkKey = DigestUtil.sha256Hex(source + "|" + chunk.getText());
            chunk.getMetadata().put("chunk_key", chunkKey);
        }
        log.info("RAG documents loaded and chunked: {} raw docs -> {} chunks from [{}]",
                raw.size(), chunks.size(), documentsPath);
        return chunks;
    }

    /**
     * 从配置路径加载所有 .md 文件，每个文件一个 Document（未分块）。
     */
    public List<Document> loadMarkdownDocuments() {
        List<Document> allDocuments = new ArrayList<>();
        try {
            Resource[] resources = resourcePatternResolver.getResources(documentsPath);
            for (Resource resource : resources) {
                if (!resource.exists() || !resource.isReadable()) {
                    log.debug("Skipping non-readable resource: {}", resource);
                    continue;
                }
                String filename = resource.getFilename();
                if (filename == null || !filename.endsWith(".md")) {
                    continue;
                }
                MarkdownDocumentReaderConfig config = MarkdownDocumentReaderConfig.builder()
                        .withHorizontalRuleCreateDocument(false)
                        .withIncludeCodeBlock(true)
                        .withIncludeBlockquote(true)
                        .withAdditionalMetadata("source", resource.getURI() != null ? resource.getURI().toString() : filename)
                        .withAdditionalMetadata("filename", filename)
                        .build();
                MarkdownDocumentReader reader = new MarkdownDocumentReader(resource, config);
                List<Document> docs = reader.get();
                for (Document doc : docs) {
                    Map<String, Object> meta = doc.getMetadata() != null ? doc.getMetadata() : new java.util.HashMap<>();
                    if (!meta.containsKey("filename")) meta.put("filename", filename);
                    if (!meta.containsKey("source")) meta.put("source", resource.getURI() != null ? resource.getURI().toString() : filename);
                    allDocuments.add(new Document(doc.getText(), meta));
                }
            }
        } catch (IOException e) {
            log.error("Failed to load Markdown documents from [{}]", documentsPath, e);
        }
        return allDocuments;
    }
}
