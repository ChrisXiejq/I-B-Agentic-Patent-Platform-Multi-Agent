package com.inovationbehavior.backend.ai.tools;

import com.inovationbehavior.backend.model.Patent;
import com.inovationbehavior.backend.service.intf.PatentService;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * 专利热度/状态工具
 */
@Component
public class PatentHeatTool {

    @Autowired
    private PatentService patentService;

    @Tool(description = "Query the heat or attention level of a patent, based on patent status and evaluation (status=1 means evaluated/high heat)")
    public String getPatentHeat(
            @ToolParam(description = "Patent number, e.g. CN or application number") String patentNo) {
        if (patentNo == null || patentNo.isBlank()) {
            return "Please provide patent number patentNo";
        }
        try {
            Patent patent = patentService.getPatentByNo(patentNo.trim());
            if (patent == null) {
                return "Patent not found: " + patentNo;
            }
            Integer status = patent.getStatus();
            String heatDesc = (status != null && status == 1)
                    ? "Evaluated, high heat"
                    : "Not evaluated or normal status";
            return String.format("Patent %s heat: %s (status=%s). Can combine patent details and survey data for further attention analysis.",
                    patent.getNo(), heatDesc, status);
        } catch (Exception e) {
            return "Failed to query patent heat: " + e.getMessage();
        }
    }
}
