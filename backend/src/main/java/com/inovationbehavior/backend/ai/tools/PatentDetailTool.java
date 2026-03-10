package com.inovationbehavior.backend.ai.tools;

import com.inovationbehavior.backend.model.Patent;
import com.inovationbehavior.backend.service.intf.PatentService;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 专利详情查询工具
 */
@Component
public class PatentDetailTool {

    @Autowired
    private PatentService patentService;

    @Tool(description = "Query patent details by patent number, including name, summary, link, type, status, PDF list, etc.")
    public String getPatentDetails(
            @ToolParam(description = "Patent number, e.g. CN or application number") String patentNo) {
        if (patentNo == null || patentNo.isBlank()) {
            return "Please provide patent number patentNo";
        }
        try {
            Patent patent = patentService.getPatentByNo(patentNo.trim());
            if (patent == null) {
                return "Patent not found: " + patentNo;
            }
            StringBuilder sb = new StringBuilder();
            sb.append("Patent No: ").append(patent.getNo()).append("\n");
            sb.append("Name: ").append(patent.getName() != null ? patent.getName() : "-").append("\n");
            sb.append("Summary: ").append(patent.getSummary() != null ? patent.getSummary() : "-").append("\n");
            sb.append("Link: ").append(patent.getLink() != null ? patent.getLink() : "-").append("\n");
            sb.append("Applicant: ").append(patent.getAppln_application() != null ? patent.getAppln_application() : "-").append("\n");
            sb.append("Type: ").append(patent.getType() != null ? patent.getType() : "-").append("\n");
            sb.append("Status: ").append(patent.getStatus() != null ? patent.getStatus() : "-").append("\n");
            sb.append("Agency: ").append(patent.getAgency() != null ? patent.getAgency() : "-").append("\n");
            if (patent.getUpdate_time() != null) {
                sb.append("Update time: ").append(patent.getUpdate_time()).append("\n");
            }
            List<String> pdfs = patent.getPdfs();
            if (pdfs != null && !pdfs.isEmpty()) {
                sb.append("PDF count: ").append(pdfs.size()).append("\n");
                sb.append("PDF list: ").append(pdfs.stream().limit(5).collect(Collectors.joining("; ")));
                if (pdfs.size() > 5) sb.append(" ...and ").append(pdfs.size()).append(" total");
            } else {
                sb.append("PDF: None");
            }
            return sb.toString();
        } catch (Exception e) {
            return "Failed to query patent details: " + e.getMessage();
        }
    }
}
