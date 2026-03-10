package com.inovationbehavior.backend.service.intf;

import com.inovationbehavior.backend.model.economic.Article;
import com.inovationbehavior.backend.model.intelligence.Course;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Service
public interface CourseService {
    Course createCourese(Course course, MultipartFile file_en, MultipartFile file_zn, MultipartFile file_de) throws IOException;

    void deleteCourse(Long id);

    void updateCourse(Course course);
}
