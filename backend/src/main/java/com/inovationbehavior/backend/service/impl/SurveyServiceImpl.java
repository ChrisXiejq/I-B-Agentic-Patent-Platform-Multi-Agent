package com.inovationbehavior.backend.service.impl;

import com.inovationbehavior.backend.mapper.PatentMapper;
import com.inovationbehavior.backend.mapper.SurveyMapper;
import com.inovationbehavior.backend.model.survey.AwardInfo;
import com.inovationbehavior.backend.model.survey.Survey;
import com.inovationbehavior.backend.service.intf.SurveyService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Service
public class SurveyServiceImpl implements SurveyService {
    @Autowired
    private SurveyMapper surveyMapper;
    @Autowired
    private PatentMapper patentMapper;
    @Override
    public void postSurvey(Survey survey) {
        surveyMapper.postSurvey(survey);
    }

    @Override
    public void postIdentification(String patentNo, String identification,String invitationCode, String type) {
        LocalDateTime time = LocalDateTime.now();
        surveyMapper.postIdentification(patentNo,identification,invitationCode, time, type);
        patentMapper.updateStatus(patentNo);
    }

    @Override
    public void postEnterprise(String patentNo, String enterprise,String invitationCode, String type) {
        LocalDateTime time = LocalDateTime.now();
        surveyMapper.postEnterprise(patentNo,enterprise,invitationCode, time, type);
        patentMapper.updateStatus(patentNo);
    }

    @Override
    public void postValue(String patentNo, String value,String invitationCode, String type) {
        LocalDateTime time = LocalDateTime.now();
        surveyMapper.postValue(patentNo,value,invitationCode, time, type);
        patentMapper.updateStatus(patentNo);
    }

    @Override
    public void postUsage(String patentNo, String usage,String invitationCode, String type) {
        LocalDateTime time = LocalDateTime.now();
        surveyMapper.postUsage(patentNo, usage,invitationCode, time, type);
        patentMapper.updateStatus(patentNo);
    }

    @Override
    public void postPolicy(String patentNo, String policy,String invitationCode, String type) {
        LocalDateTime time = LocalDateTime.now();
        surveyMapper.postPolicy(patentNo, policy, invitationCode,time, type);
        patentMapper.updateStatus(patentNo);
    }

    @Override
    public int postAwardInfo(AwardInfo requestData) {
        System.out.println(requestData);
        System.out.println(surveyMapper.checkAward(requestData.getPatentNo()));
        if(surveyMapper.checkAward(requestData.getPatentNo())==0){
            surveyMapper.postAwardInfo(requestData.getPatentNo(), requestData.getAward(),requestData.getAddress(), requestData.getPostCode(), requestData.getReceiverName(), requestData.getPhoneNumber());
            return 1;
        }
        return 0;
    }
}
