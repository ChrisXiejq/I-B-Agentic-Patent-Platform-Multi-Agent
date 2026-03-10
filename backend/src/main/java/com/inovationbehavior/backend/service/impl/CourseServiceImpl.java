package com.inovationbehavior.backend.service.impl;

import com.inovationbehavior.backend.mapper.CourseMapper;
import com.inovationbehavior.backend.model.economic.Article;
import com.inovationbehavior.backend.model.intelligence.Course;
import com.inovationbehavior.backend.service.intf.CourseService;
import com.inovationbehavior.backend.util.PutFile;
import com.inovationbehavior.backend.util.TemporaryCredentialsProvider;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.TreeMap;

@Service
@Slf4j
public class CourseServiceImpl implements CourseService {

    @Autowired
    CourseMapper courseMapper;
    @Override
    public Course createCourese(Course course, MultipartFile file_en, MultipartFile file_zn,MultipartFile file_de) throws IOException {
        //此时的course没有保存url
        courseMapper.insertCourse(course);
        // 更新文件名（如果文件存在）
        course.setUrl_en(file_en != null ? course.get_id() + file_en.getOriginalFilename() : null);
        course.setUrl_zn(file_zn != null ? course.get_id() + file_zn.getOriginalFilename() : null);
        course.setUrl_de(file_de != null ? course.get_id() + file_de.getOriginalFilename() : null);

        //更新course
        courseMapper.updateAllCourse(course);
//        course.set_id(courseId);
        if (course.getLecturers() != null && !course.getLecturers().isEmpty()) {
            courseMapper.insertLecturers(course.getLecturers(), course.get_id());
        }
        // 获取临时密钥
        TemporaryCredentialsProvider credentialsProvider = new TemporaryCredentialsProvider();
        TreeMap<String, Object> credentials = credentialsProvider.getTemporaryCredentials();

        // 实例化文件上传类
        PutFile putFile = new PutFile("course");
        // 遍历并上传每个文件
        MultipartFile[] files = new MultipartFile[3];
        files[0] = file_en;files[1] = file_zn;files[2] = file_de;
        for (MultipartFile file : files) {
            if(file!=null) {
                // 构造新的文件名，加上courseId作为前缀
                String originalFilename = file.getOriginalFilename();
                String newFilename = course.get_id() + (originalFilename != null ? originalFilename : "");
                File convFile = new File(System.getProperty("java.io.tmpdir") + "/" + newFilename);
                file.transferTo(convFile);

                // 上传文件到COS
                putFile.PutFileToCOS(credentials, convFile, newFilename);
            }
        }
        return course;
    }

    @Override
    public void deleteCourse(Long id) {
        courseMapper.deleteCourse(id);
    }

    @Override
    public void updateCourse(Course course) {
        courseMapper.updateCourse(course);
    }
}
