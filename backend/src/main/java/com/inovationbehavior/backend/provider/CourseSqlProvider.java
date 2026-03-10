package com.inovationbehavior.backend.provider;

import com.inovationbehavior.backend.model.intelligence.Course;
import org.apache.ibatis.jdbc.SQL;

public class CourseSqlProvider {
    public String updateCourseSql(final Course course) {
        return new SQL() {{
            UPDATE("course");

            if (course.getType() != null) {
                SET("type = #{course.type}");
            }
            if (course.getTitle_en() != null) {
                SET("title_en = #{course.title_en}");
            }
            if (course.getTitle_zn() != null) {
                SET("title_zn = #{course.title_zn}");
            }
            if (course.getTitle_de() != null) {
                SET("title_de = #{course.title_de}");
            }

            WHERE("_id = #{course._id}");
        }}.toString();
    }
}
