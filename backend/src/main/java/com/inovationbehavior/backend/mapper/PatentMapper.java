package com.inovationbehavior.backend.mapper;

import com.inovationbehavior.backend.model.Patent;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.sql.Timestamp;
import java.util.List;
import java.util.Map;

@Mapper
public interface PatentMapper {
    @Select("""
    SELECT appln_ID no, title_zn name,abstract_zn summary, link,update_time,type,status FROM patent
    WHERE appln_ID = #{No}
    AND exist = 1
    """)
    Patent getPatentByNo(String No);

    @Select("""
        SELECT publn_nr FROM patent_publn
        WHERE appln_id = #{No}
    """)
    List<String> getPatentPdfs(String No);

    @Select("""
    SELECT DISTINCT appln_application application FROM patent 
    WHERE appln_application LIKE CONCAT('%',#{key},'%')
    """)
    List<String> getCompanyByKey(String Key);

    @Select("""
SELECT DISTINCT appln_ID no, title_zn name, appln_application, type, status FROM patent
WHERE exist = 1
AND appln_application REGEXP CONCAT('^.*', #{regexKey}, '.*$')
""")
    List<Patent> getPatentsByKey(String regexKey);

    @Select("""
SELECT DISTINCT p.appln_ID no, title_zn name, appln_application, p.type, status,pa.agency FROM patent p
LEFT JOIN patent_agency pa on p.appln_ID = pa.appln_id
WHERE p.exist = 1
AND pa.agency REGEXP CONCAT('^.*', #{regexKey}, '.*$')
""")
    List<Patent> getPatentsByKeyFromAgency(String regexKey);

    @Update(
            """
    UPDATE patent, survey
    SET patent.status = 1
    where patent.appln_ID = #{patentNo}
    AND survey.patent_no = #{patentNo}
    AND survey.value is not null
    AND survey.use is not null
"""
    )
    void updateStatus(String patentNo);

    @Select("""
    SELECT appln_ID no, title_zn name,abstract_zn summary FROM patent
    WHERE exist = 1
    """)
    List<Patent> getAllExistPatents();

    @Update("UPDATE patent SET status = 0 WHERE patent.appln_ID = #{patentNo}")
    void refreshStatus(String patentNo);
}
