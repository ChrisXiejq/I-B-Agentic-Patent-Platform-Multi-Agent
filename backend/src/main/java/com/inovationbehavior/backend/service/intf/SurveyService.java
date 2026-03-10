package com.inovationbehavior.backend.service.intf;

import com.inovationbehavior.backend.model.survey.AwardInfo;
import com.inovationbehavior.backend.model.survey.Survey;

public interface SurveyService {
    void postSurvey(Survey survey);

    void postIdentification(String patentNo, String identification,String invitationCode, String type);

    void postEnterprise(String patentNo, String enterprise,String invitationCode, String type);

    void postValue(String patentNo, String value,String invitationCode, String type);

    void postUsage(String patentNo, String usage,String invitationCode, String type);

    void postPolicy(String patentNo, String policy,String invitationCode, String type);

    int postAwardInfo(AwardInfo requestData);
}
