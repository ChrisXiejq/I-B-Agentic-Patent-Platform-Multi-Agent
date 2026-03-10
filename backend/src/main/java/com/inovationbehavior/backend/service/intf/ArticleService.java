package com.inovationbehavior.backend.service.intf;

import com.inovationbehavior.backend.model.economic.Article;
import com.inovationbehavior.backend.model.intelligence.Course;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Service
public interface ArticleService {
    Article createArticle(Article article);
    void deleteArticle(Long id);

    void updateArticle(Article article);
}
