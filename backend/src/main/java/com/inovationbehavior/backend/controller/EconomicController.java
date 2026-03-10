package com.inovationbehavior.backend.controller;

import com.inovationbehavior.backend.model.Result;
import com.inovationbehavior.backend.service.intf.EconomicService;
import com.inovationbehavior.backend.service.intf.IntelligenceService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/economic")
public class EconomicController {
    @Autowired
    EconomicService economicService;

    @GetMapping("/articles")
    public Result getArticleByPage(@RequestParam(defaultValue = "1",required = false) Integer page,
                                  @RequestParam(defaultValue = "10",required = false) Integer size,
                                   @RequestParam() String type)
    {
        return Result.success(economicService.getArticleByPage(page,size,type));
    }

    @GetMapping("/articles/{id}")
    public Result getArticleByPage(@PathVariable("id") Long id)
    {
        return Result.success(economicService.getArticleById(id));
    }

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
}
