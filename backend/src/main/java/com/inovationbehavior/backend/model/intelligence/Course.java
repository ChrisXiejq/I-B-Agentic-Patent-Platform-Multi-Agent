package com.inovationbehavior.backend.model.intelligence;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * 通用课程类
 * 用于存储需要标题、作者、docx文章链接的课程信息。
 *
 * @author litianlololo
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class Course{
    Long _id;
    String type;
    String title_en;
    String title_zn;
    String title_de;
    String url_en;
    String url_zn;
    String url_de;
    List<Lecturer> lecturers;
}
