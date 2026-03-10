package com.inovationbehavior.backend.service.intf;

import com.inovationbehavior.backend.model.news.IbNews;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

public interface IbNewsService {
    IbNews getNewsById(Integer id);

    List<IbNews> getNewsList();

    IbNews createNews(IbNews ibNews, MultipartFile pdfFile, MultipartFile imageFile) throws IOException;

    void deleteNewsById(Integer id);
}
