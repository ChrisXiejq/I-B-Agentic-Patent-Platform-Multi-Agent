package com.inovationbehavior.backend.ai.tenant;

/**
 * 当前请求的租户 ID，用于 RAG 检索时按文档 tenant_id 过滤（文档级访问控制）。
 */
public final class TenantContext {

    private static final ThreadLocal<String> HOLDER = new ThreadLocal<>();

    public static void setTenantId(String tenantId) {
        HOLDER.set(tenantId);
    }

    public static String getTenantId() {
        return HOLDER.get();
    }

    public static void clear() {
        HOLDER.remove();
    }
}
