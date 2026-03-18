package com.inovationbehavior.backend.ai.tenant;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

/**
 * 从请求头 X-Tenant-Id 读取租户 ID 并写入 TenantContext，供 RAG 检索过滤；请求结束后清除。
 */
@Component
@Order(1)
@ConditionalOnProperty(name = "app.rag.tenant.enabled", havingValue = "true")
public class TenantContextFilter extends OncePerRequestFilter {

    public static final String HEADER_TENANT_ID = "X-Tenant-Id";

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {
        try {
            String tenantId = request.getHeader(HEADER_TENANT_ID);
            if (tenantId != null && !tenantId.isBlank()) {
                TenantContext.setTenantId(tenantId.trim());
            }
            filterChain.doFilter(request, response);
        } finally {
            TenantContext.clear();
        }
    }
}
