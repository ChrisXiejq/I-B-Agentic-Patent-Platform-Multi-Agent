package com.inovationbehavior.backend.controller;

import com.inovationbehavior.backend.annotation.LogAnnotation;
import com.inovationbehavior.backend.model.Patent;
import com.inovationbehavior.backend.model.Result;
import com.inovationbehavior.backend.service.intf.PatentService;
import jakarta.websocket.server.PathParam;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/patents")
public class PatentController {
    @Autowired
    PatentService patentService;

    @LogAnnotation("根据No号查询详情")
    @GetMapping("/{No}")
    public Result getPatentByNo(@PathVariable String No) {
        Patent tmp = patentService.getPatentByNo(No);
        if (tmp != null) {
            return Result.success(tmp);
        }
        return Result.error("未找到No：" + No);
    }
    @LogAnnotation("根据key范围查询专利号")
    @GetMapping("")
    public Result getPatentByKey(@PathParam("company") String company) {

        return Result.success( patentService.getPatentByKey(company));
    }
    @LogAnnotation("根据Company范围查询公司名")
    @GetMapping("/company")
    public Result getCompanyByKey(@PathParam("key") String key){
        System.out.println(key);
        return Result.success(patentService.getCompanyByKey(key));
    }

    @LogAnnotation("开始无可用pdf论文统计")
    @GetMapping("/pdfs")
    public Result getAllPatentIdWithoutPdfs() throws IOException {
        patentService.getAllPatentIdWithoutPdfs();
        return  Result.success();
    }

    @LogAnnotation("清空填写记录")
    @GetMapping("/clear/{No}")
    public Result clearPatentByNo(@PathVariable String No) {
        return Result.success(patentService.clearPatentByNo(No));
    }
}
