package com.inovationbehavior.backend.mapper;

import com.inovationbehavior.backend.model.survey.AwardInfo;
import com.inovationbehavior.backend.model.survey.Survey;
import org.apache.ibatis.annotations.*;

import java.time.LocalDateTime;

@Mapper
public interface SurveyMapper {
    @Insert("insert into survey (patent_no, identification, enterprise, value) values (#{patentNo},#{identification},#{evaluate},#{value})")
    void postSurvey(Survey survey);

    @Insert("INSERT INTO survey (patent_no, identification,invitationCode ,time, type) \n" +
            "VALUES (#{patentNo}, #{identification},#{invitationCode}, #{time}, #{type} ) \n" +
            "ON DUPLICATE KEY UPDATE identification = VALUES(identification), time = VALUES(time), type = VALUES(type);")
    void postIdentification(String patentNo, String identification,String invitationCode, LocalDateTime time, String type);

    @Insert("INSERT INTO survey (patent_no, enterprise,invitationCode ,time, type) \n" +
            "VALUES (#{patentNo}, #{enterprise},#{invitationCode}, #{time}, #{type}) \n" +
            "ON DUPLICATE KEY UPDATE enterprise = VALUES(enterprise), time = VALUES(time), type = VALUES(type);")
    void postEnterprise(String patentNo, String enterprise,String invitationCode, LocalDateTime time, String type);

    @Insert("INSERT INTO survey (patent_no, value,invitationCode, time, type) \n" +
            "VALUES (#{patentNo}, #{value},#{invitationCode}, #{time}, #{type}) \n" +
            "ON DUPLICATE KEY UPDATE value = VALUES(value), time = VALUES(time), type = VALUES(type);")
    void postValue(String patentNo, String value,String invitationCode, LocalDateTime time, String type);

    @Insert("INSERT INTO survey (patent_no, `use`,invitationCode, time, type) \n" +
            "VALUES (#{patentNo}, #{use},#{invitationCode}, #{time}, #{type}) \n" +
            "ON DUPLICATE KEY UPDATE `use` = VALUES(`use`), time = VALUES(time), type = VALUES(type);")
    void postUsage(String patentNo, String use,String invitationCode, LocalDateTime time, String type);


    @Insert("INSERT INTO survey (patent_no, policy,invitationCode, time, type) \n" +
            "VALUES (#{patentNo}, #{policy},#{invitationCode}, #{time}, #{type}) \n" +
            "ON DUPLICATE KEY UPDATE policy = VALUES(policy), time = VALUES(time), type = VALUES(type);")
    void postPolicy(String patentNo, String policy,String invitationCode, LocalDateTime time, String type);

    @Insert("INSERT INTO survey (patent_no, award, address, post_code, receiver_name, phone_number) " +
            "VALUES (#{patentNo}, #{award}, #{address}, #{postCode}, #{receiverName}, #{phoneNumber}) " +
            "ON DUPLICATE KEY UPDATE award = IFNULL(award, VALUES(award)), address = VALUES(address), post_code = VALUES(post_code), receiver_name = VALUES(receiver_name), phone_number = VALUES(phone_number);")
    int postAwardInfo(String patentNo, String award, String address, String postCode, String receiverName, String phoneNumber);

    @Select("SELECT COUNT(*) FROM survey WHERE patent_no = #{patentNo} AND award IS NOT NULL")
    int checkAward(String patentNo);

    @Select("SELECT * FROM survey WHERE patent_no = #{patentNo}")
    Survey getSurvey(String patentNo);

    @Delete("DELETE FROM survey WHERE patent_no = #{patentNo}")
    void deleteSurvey(String patentNo);

    @Insert("""
    INSERT INTO survey_copy (patent_no, identification, enterprise, value, `use`, policy, award, address, time, invitationCode, type)
    SELECT patent_no, identification, enterprise, value, `use`, policy, award, address, time, invitationCode, type
    FROM survey
    WHERE patent_no = #{patentNo}
    ON DUPLICATE KEY UPDATE
        identification = VALUES(identification),
        enterprise = VALUES(enterprise),
        value = VALUES(value),
        `use` = VALUES(`use`),
        policy = VALUES(policy),
        award = VALUES(award),
        address = VALUES(address),
        time = VALUES(time),
        invitationCode = VALUES(invitationCode),
        type = VALUES(type)
""")
    void copySurvey(String patentNo);

}
