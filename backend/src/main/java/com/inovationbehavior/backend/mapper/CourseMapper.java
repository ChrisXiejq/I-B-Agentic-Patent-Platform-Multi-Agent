package com.inovationbehavior.backend.mapper;

import com.inovationbehavior.backend.model.intelligence.Course;
import com.inovationbehavior.backend.model.intelligence.Lecturer;
import com.inovationbehavior.backend.provider.CourseSqlProvider;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface CourseMapper {
    @Insert("INSERT INTO course (type, title_en, title_zn, title_de, url_en, url_zn, url_de) " +
            "VALUES (#{course.type}, #{course.title_en}, #{course.title_zn}, " +
            "#{course.title_de}, #{course.url_en}, #{course.url_zn}, #{course.url_de})")
    @Options(useGeneratedKeys = true, keyProperty = "course._id")
    Long insertCourse(@Param("course") Course course);

    @Update("UPDATE course SET " +
            "type = #{course.type}, " +
            "title_en = #{course.title_en}, " +
            "title_zn = #{course.title_zn}, " +
            "title_de = #{course.title_de}, " +
            "url_en = #{course.url_en}, " +
            "url_zn = #{course.url_zn}, " +
            "url_de = #{course.url_de} " +
            "WHERE _id = #{course._id}")
    int updateAllCourse(@Param("course") Course course);

    @UpdateProvider(type = CourseSqlProvider.class, method = "updateCourseSql")
    int updateCourse(@Param("course") Course course);

    @Insert({
            "<script>",
            "INSERT INTO course_lecturer (name_en, name_zn, name_de, course_id) VALUES ",
            "<foreach collection='lecturers' item='lecturer' index='index' separator=','>",
            "(#{lecturer.name_en}, #{lecturer.name_zn}, #{lecturer.name_de}, #{courseId})",
            "</foreach>",
            "</script>"
    })
    int insertLecturers(@Param("lecturers") List<Lecturer> lecturers, @Param("courseId") Long courseId);

    @Delete("""
        DELETE FROM course
        WHERE _id = #{course_id};
    """)
    void deleteCourse(@Param("course_id") Long course_id);
}
