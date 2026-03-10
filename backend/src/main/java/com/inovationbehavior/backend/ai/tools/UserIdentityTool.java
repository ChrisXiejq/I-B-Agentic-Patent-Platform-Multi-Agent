package com.inovationbehavior.backend.ai.tools;

import com.inovationbehavior.backend.mapper.SurveyMapper;
import com.inovationbehavior.backend.model.survey.Survey;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * 用户身份相关工具（与问卷/邀请码关联）
 */
@Component
public class UserIdentityTool {

    @Autowired(required = false)
    private SurveyMapper surveyMapper;

    @Tool(description = "Query user/survey identity by invitation code or patent number, for judging the association between current user and patent survey")
    public String getUserIdentity(
            @ToolParam(description = "Invitation code, optional") String invitationCode,
            @ToolParam(description = "Patent number, optional, for querying survey identity under this patent") String patentNo) {
        if (surveyMapper == null) {
            return "User identity service temporarily unavailable (SurveyMapper not injected)";
        }
        if (patentNo != null && !patentNo.isBlank()) {
            Survey survey = surveyMapper.getSurvey(patentNo);
            if (survey == null) {
                return "No survey/user identity record under patent " + patentNo;
            }
            return String.format("Survey identity for patent %s: identification=%s, enterprise=%s, value=%s, use=%s, policy=%s, invitationCode=%s",
                    patentNo,
                    survey.getIdentification() != null ? survey.getIdentification() : "-",
                    survey.getEnterprise() != null ? survey.getEnterprise() : "-",
                    survey.getValue() != null ? survey.getValue() : "-",
                    survey.getUsage() != null ? survey.getUsage() : "-",
                    survey.getPolicy() != null ? survey.getPolicy() : "-",
                    survey.getInvitationCode() != null ? survey.getInvitationCode() : "-");
        }
        if (invitationCode != null && !invitationCode.isBlank()) {
            return "Current user identified by invitation code: " + invitationCode + ". This code is used for survey submission and identity association; specific survey content requires patent number to query.";
        }
        return "invitationCode or patentNo not provided. Pass patentNo to query survey identity under that patent, or pass invitationCode to indicate current user identity.";
    }
}
