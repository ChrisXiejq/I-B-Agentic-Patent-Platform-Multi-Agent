package com.inovationbehavior.backend.ai.rag.retrieval;

import com.inovationbehavior.backend.ai.tenant.TenantContext;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * 文档级访问控制：按当前租户过滤检索结果。文档 metadata 中 tenant_id 为空或与当前租户一致时保留。
 */
public class TenantAwareDocumentRetriever implements DocumentRetriever {

    private final DocumentRetriever delegate;
    /** 未设 tenant 时是否返回全部文档（true）还是空（false） */
    private final boolean noTenantMeansAll;

    public TenantAwareDocumentRetriever(DocumentRetriever delegate, boolean noTenantMeansAll) {
        this.delegate = delegate;
        this.noTenantMeansAll = noTenantMeansAll;
    }

    @Override
    public List<Document> retrieve(Query query) {
        List<Document> docs = delegate.retrieve(query);
        String tenantId = TenantContext.getTenantId();
        if (tenantId == null || tenantId.isBlank()) {
            return noTenantMeansAll ? docs : List.of();
        }
        return docs.stream()
                .filter(d -> {
                    Object docTenant = d.getMetadata() != null ? d.getMetadata().get("tenant_id") : null;
                    if (docTenant == null || (docTenant instanceof String s && s.isBlank())) {
                        return true;
                    }
                    return Objects.equals(String.valueOf(docTenant).trim(), tenantId.trim());
                })
                .collect(Collectors.toList());
    }
}
