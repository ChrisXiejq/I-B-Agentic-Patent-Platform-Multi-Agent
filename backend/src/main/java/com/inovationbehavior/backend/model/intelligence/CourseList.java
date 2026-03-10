package com.inovationbehavior.backend.model.intelligence;

import com.inovationbehavior.backend.model.intelligence.Course;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CourseList {

    List<Course> lists;
    Long count;
}
