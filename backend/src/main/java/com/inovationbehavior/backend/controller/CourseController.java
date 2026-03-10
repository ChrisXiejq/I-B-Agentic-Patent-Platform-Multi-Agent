package com.inovationbehavior.backend.controller;

import com.inovationbehavior.backend.model.Result;
import com.inovationbehavior.backend.model.economic.Article;
import com.inovationbehavior.backend.model.intelligence.Course;
import com.inovationbehavior.backend.service.intf.CourseService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/courses")
public class CourseController {
    @Autowired
    CourseService courseService;
    @PostMapping("")
    public Result createCourese(@RequestPart("course") Course course,
                                @RequestPart(name = "file_en", required = false) MultipartFile file_en,
                                @RequestPart(name = "file_zn", required = false) MultipartFile file_zn,
                                @RequestPart(name = "file_de", required = false) MultipartFile file_de) throws IOException {
        return Result.success(courseService.createCourese(course, file_en, file_zn, file_de));
    }
    @DeleteMapping("/{id}")
    public Result deleteArticle(@PathVariable Long id){
        courseService.deleteCourse(id);
        return Result.success();
    }

    @PutMapping("")
    public Result updateArticle(@RequestPart("course") Course course){
        courseService.updateCourse(course);
        return Result.success();
    }
}
