package com.inovationbehavior.backend.mapper;

import com.inovationbehavior.backend.model.intelligence.Course;
import com.inovationbehavior.backend.model.intelligence.Lecturer;
import com.inovationbehavior.backend.model.intelligence.PPAuthor;
import com.inovationbehavior.backend.model.intelligence.PatentPledge;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface IntelligenceMapper {
    @Select("""
    SELECT count(*) FROM course
    WHERE type = #{type}
    """)
    Long courseCount(@Param("type") String type);
    @Select("""
        SELECT _id, title_en,title_zn,title_de, url_en, url_zn, url_de
        FROM course
        WHERE type = #{type}
        LIMIT #{start},#{size}
    """)
    List<Course> getCourseByPage(@Param("start") Integer start, @Param("size") Integer size, @Param("type") String type);

    @Select("""
        SELECT _id, name_en, name_zn, name_de, course_id
        FROM course_lecturer
        WHERE course_id = #{course_id}
    """)
    List<Lecturer> getLecturersByCourseId(@Param("course_id") Long course_id);

    @Select("""
    SELECT * FROM course
    WHERE _id = #{course_id}
""")
    Course getCourseById(@Param("course_id") Long course_id);
}
//    @Select("""
//    SELECT count(*) FROM patentpledge
//    """)
//    Long patentPledgeCount();
//    @Select("""
//        SELECT _id, title_en,title_zn,title_de, url_en, url_zn, url_de
//        FROM patentpledge
//        LIMIT #{start},#{size}
//    """)
//    List<PatentPledge> getPatentPledgeByPage(@Param("start") Integer start, @Param("size") Integer size);
//
//    @Select("""
//        SELECT _id, name_en, name_zn, name_de, patentpledge_id
//        FROM patentpledge_author
//        WHERE patentpledge_id = #{patentpledge_id}
//    """)
//    List<PPAuthor> getPPAuthorByPatentPledgeId(@Param("patentpledge_id") Long patentpledge_id);
//
//    @Select("""
//    SELECT * FROM patentpledge
//    WHERE _id = #{patentpledge_id}
//""")
//    PatentPledge getPatentPledgeById(@Param("patentpledge_id") Long patentpledge_id);

