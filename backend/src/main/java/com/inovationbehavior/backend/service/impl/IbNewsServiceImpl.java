package com.inovationbehavior.backend.service.impl;

import com.inovationbehavior.backend.mapper.IbNewsMapper;
import com.inovationbehavior.backend.model.news.IbNews;
import com.inovationbehavior.backend.service.intf.IbNewsService;
import com.inovationbehavior.backend.util.PutFile;
import com.inovationbehavior.backend.util.TemporaryCredentialsProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Date;
import java.util.List;
import java.util.TreeMap;

@Service
public class IbNewsServiceImpl implements IbNewsService {
    @Autowired
    private IbNewsMapper ibNewsMapper;
    @Override
    public IbNews getNewsById(Integer id) {
        return ibNewsMapper.getNewsById(id);
    }

    @Override
    public List<IbNews> getNewsList() {
        return ibNewsMapper.getNewsList();
    }

    @Override
    public IbNews createNews(IbNews ibNews, MultipartFile pdfFile, MultipartFile imageFile) throws IOException {
        ibNews.setTime(new Date());

        String COSPath = "https://inovation-behavior-1324093188.cos.ap-shanghai.myqcloud.com/news/";
        // 获取临时密钥
        TemporaryCredentialsProvider credentialsProvider = new TemporaryCredentialsProvider();
        TreeMap<String, Object> credentials = credentialsProvider.getTemporaryCredentials();

        // 实例化文件上传类
        PutFile putFile = new PutFile("news");

        // 构造新的pdf文件名
        String newPdfFilename = ibNews.getTime() + ibNews.getTitleZn()+pdfFile.getOriginalFilename();
        File convPdfFile = new File(System.getProperty("java.io.tmpdir") + "/" + newPdfFilename);
        pdfFile.transferTo(convPdfFile);

        // 上传文件到COS
        putFile.PutFileToCOS(credentials, convPdfFile, newPdfFilename);

        // 构造新的img文件名
        String newImgFilename = ibNews.getTime() + ibNews.getTitleZn()+imageFile.getOriginalFilename();
        File convImgFile = new File(System.getProperty("java.io.tmpdir") + "/" + newImgFilename);
        imageFile.transferTo(convImgFile);

        // 上传文件到COS
        putFile.PutFileToCOS(credentials, convImgFile, newImgFilename);

        ibNews.setUrlZn(COSPath+newPdfFilename);
        ibNews.setCover(COSPath+newImgFilename);
        ibNews.setTime(new Date());
        ibNewsMapper.insertIbNews(ibNews);
        return ibNews;
    }

    @Override
    public void deleteNewsById(Integer id) {
        ibNewsMapper.deleteNewsById(id);
    }
}
