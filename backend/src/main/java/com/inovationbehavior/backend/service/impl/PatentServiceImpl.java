package com.inovationbehavior.backend.service.impl;

import com.google.gson.Gson;
import com.inovationbehavior.backend.constants.CosConstants;
import com.inovationbehavior.backend.mapper.PatentMapper;
import com.inovationbehavior.backend.mapper.SurveyMapper;
import com.inovationbehavior.backend.model.Patent;
import com.inovationbehavior.backend.model.Result;
import com.inovationbehavior.backend.model.survey.Survey;
import com.inovationbehavior.backend.service.intf.IntelligenceService;
import com.inovationbehavior.backend.service.intf.PatentService;
import lombok.extern.slf4j.Slf4j;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import static com.inovationbehavior.backend.constants.CosConstants.PdfUrl;

@Service
@Slf4j
public class PatentServiceImpl implements PatentService {
    @Autowired
    private PatentMapper patentMapper;

    @Autowired
    private SurveyMapper surveyMapper;

    @Autowired
    private RedisTemplate<String, String> redisTemplate;

    @Autowired
    private RestTemplate restTemplate;


    @Override
    public Patent getPatentByNo(String no){
        // Redis的key
        String redisKey = "patent:" + no;

        // 尝试从Redis中获取记录
        String patentJson = redisTemplate.opsForValue().get(redisKey);
//        String patentJson = "";
        Patent patent = null;
        Gson gson = new Gson();
        if (patentJson != null && !patentJson.isEmpty()) {
            // 如果Redis中有记录，反序列化后返回
//            System.out.println("走了Redis");
            patent = gson.fromJson(patentJson, Patent.class);
        } else {
            // Redis中没有记录，查询数据库
            patent = patentMapper.getPatentByNo(no);
            if (patent != null) {
                //加入pdf地址
                List<String> pdfs = patentMapper.getPatentPdfs(no);
                System.out.println("pdfs: " + pdfs.size());
                List<String> pdfsWithPrefix = new ArrayList<>();
                for(String pdf : pdfs) {
                    String formattedPdf = PdfUrl + pdf.replace("\r", "") + ".pdf"; // 删除"/r"，并添加".pdf"后缀
                    if (canDownloadPDF(formattedPdf)) {
                        pdfsWithPrefix.add(formattedPdf);
                    }
                }
                patent.setPdfs(pdfsWithPrefix);
                // 数据库中查询到了记录，序列化后存储到Redis中
                redisTemplate.opsForValue().set(redisKey, gson.toJson(patent));
//                System.out.println("走了DB");
            }else{
                //设置null值
                redisTemplate.opsForValue().set(redisKey,gson.toJson(null));
                redisTemplate.expire(redisKey,24, TimeUnit.HOURS);
            }
        }

        return patent;
    }

    private boolean canDownloadPDF(String url) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.set("Accept", "application/pdf");
            HttpEntity<String> entity = new HttpEntity<>(headers);
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.HEAD, entity, String.class);
            System.out.println(url);
            return response.getStatusCode() == HttpStatus.OK
                    && response.getHeaders().getContentType() != null
                    && response.getHeaders().getContentType().includes(MediaType.APPLICATION_PDF);
        } catch (Exception e) {
            System.out.println("Error checking PDF URL: " + url);
            log.error("Error checking PDF URL: " + url, e);
            return false;
        }
    }

    @Override
    public List<Patent> getPatentByKey(String key) {
        // 将 key 转换为正则表达式模式
        String regexKey = key.chars()
                .mapToObj(c -> (char) c + ".*")
                .collect(Collectors.joining("", "^.*", ".*$"));
        Set<Patent> result = new HashSet<>(patentMapper.getPatentsByKey(regexKey));
        result.addAll(patentMapper.getPatentsByKeyFromAgency(regexKey));

        return new ArrayList<>(result);
    }


    @Override
    public List<String> getCompanyByKey(String key) {
        List<String> companies = patentMapper.getCompanyByKey(key);
        System.out.print(companies+key);
        return companies;
    }

    @Override
    public void getAllPatentIdWithoutPdfs() throws IOException {
        List<Patent> patents = patentMapper.getAllExistPatents();
        Map<Patent, String> patentMap = new HashMap<>();

        for (Patent patent : patents) {
            List<String> pdfs = patentMapper.getPatentPdfs(patent.getNo());
            if (pdfs.isEmpty()) {
                patentMap.put(patent, "没有映射关系");
                continue;
            }
            List<String> pdfsWithPrefix = new ArrayList<>();
            for (String pdf : pdfs) {
                String formattedPdf = CosConstants.PdfUrl + pdf.replace("\r", "") + ".pdf";
                if (canDownloadPDF(formattedPdf)) {
                    pdfsWithPrefix.add(formattedPdf);
                }
            }
            if (pdfsWithPrefix.isEmpty()) {
                patentMap.put(patent, "没有文件");
            }
        }

        // 创建 Excel 文件
        Workbook workbook = new XSSFWorkbook();
        Sheet sheet = workbook.createSheet("Patents");
        int rowIndex = 0;

        // 创建表头
        Row headerRow = sheet.createRow(rowIndex++);
        headerRow.createCell(0).setCellValue("专利编号");
        headerRow.createCell(1).setCellValue("专利名称");
        headerRow.createCell(2).setCellValue("说明");

        // 填充数据
        for (Map.Entry<Patent, String> entry : patentMap.entrySet()) {
            Row row = sheet.createRow(rowIndex++);
            row.createCell(0).setCellValue(entry.getKey().getNo());
            row.createCell(1).setCellValue(entry.getKey().getName());
            row.createCell(2).setCellValue(entry.getValue());
        }

        // 写入文件
        try (FileOutputStream outputStream = new FileOutputStream("Patents_Status.xlsx")) {
            workbook.write(outputStream);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        workbook.close();
    }

    @Override
    public Result clearPatentByNo(String no) {
        surveyMapper.copySurvey(no);
        surveyMapper.deleteSurvey(no);
        patentMapper.refreshStatus(no);
        return Result.success();
    }
}
