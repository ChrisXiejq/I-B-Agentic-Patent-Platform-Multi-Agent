package com.inovationbehavior.backend.controller;

import com.inovationbehavior.backend.model.Result;
import com.inovationbehavior.backend.model.intelligence.Course;
import com.inovationbehavior.backend.service.intf.IntelligenceService;
import com.inovationbehavior.backend.service.intf.PatentService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/intelligence")
public class IntelligenceController {
    @Autowired
    IntelligenceService intelligenceService;

    @GetMapping("/courses")
    public Result getCourseByPage(@RequestParam(defaultValue = "1",required = false) Integer page,
                                  @RequestParam(defaultValue = "10",required = false) Integer size,
                                  @RequestParam() String type)
    {
        return Result.success(intelligenceService.getCourseByPage(page,size,type));
    }

    @GetMapping("/courses/{id}")
    public Result getCourseByPage(@PathVariable("id") Long id)
    {
        return Result.success(intelligenceService.getCourseById(id));
    }

//    @GetMapping("/patentpledge")
//    public Result getPatentPledgeByPage(@RequestParam(defaultValue = "1",required = false) Integer page,
//                                  @RequestParam(defaultValue = "10",required = false) Integer size)
//    {
//        return Result.success(intelligenceService.getPatentPledgeByPage(page,size));
//    }
//
//    @GetMapping("/patentpledge/{id}")
//    public Result getPatentPledgeByPage(@PathVariable("id") Long id)
//    {
//        return Result.success(intelligenceService.getPatentPledgeById(id));
//    }
}
