package com.inovationbehavior.backend.controller;

import com.inovationbehavior.backend.model.Result;
import com.inovationbehavior.backend.model.news.IbNews;
import com.inovationbehavior.backend.service.intf.IbNewsService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/ibnews")
public class IbNewsController {
    @Autowired
    private IbNewsService IbNewsService;

    @GetMapping("")
    public Result getNewsList(){
        return Result.success(IbNewsService.getNewsList());
    }

    @GetMapping("singleNews/{id}")
    public Result getNewsById(@PathVariable("id") Integer id){
        return Result.success(IbNewsService.getNewsById(id));
    }

    @PostMapping("")
    public Result createNews(@RequestPart("title_zn") String title_zn,
                             @RequestPart("pdfFile") MultipartFile pdfFile,
                             @RequestPart("imageFile") MultipartFile imageFile) throws IOException {
        IbNews ibNews = new IbNews(); ibNews.setTitleZn(title_zn);
        return Result.success(IbNewsService.createNews(ibNews,pdfFile,imageFile));}

    @DeleteMapping("/{id}")
    public Result deleteNewsById(@PathVariable("id") Integer id){IbNewsService.deleteNewsById(id); return Result.success();}
}
