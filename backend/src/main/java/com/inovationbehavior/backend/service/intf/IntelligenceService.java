package com.inovationbehavior.backend.service.intf;

import com.inovationbehavior.backend.model.intelligence.Course;
import com.inovationbehavior.backend.model.intelligence.CourseList;
import com.inovationbehavior.backend.model.intelligence.PatentPledge;
import com.inovationbehavior.backend.model.intelligence.PatentPledgeList;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Service
public interface IntelligenceService {
    CourseList getCourseByPage(Integer page, Integer size, String type);

    Course getCourseById(Long id);
}
