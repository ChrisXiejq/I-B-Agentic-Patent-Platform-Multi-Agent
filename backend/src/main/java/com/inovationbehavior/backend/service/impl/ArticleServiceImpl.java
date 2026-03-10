package com.inovationbehavior.backend.service.impl;

import com.inovationbehavior.backend.mapper.ArticleMapper;
import com.inovationbehavior.backend.model.economic.Article;
import com.inovationbehavior.backend.service.intf.ArticleService;
import com.inovationbehavior.backend.util.PutFile;
import com.inovationbehavior.backend.util.TemporaryCredentialsProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.util.TreeMap;

@Service
public class ArticleServiceImpl implements ArticleService {

    @Autowired
    ArticleMapper articleMapper;

    @Override
    public Article createArticle(Article article) {
        articleMapper.insertArticle(article);

        if (article.getAuthors() != null && !article.getAuthors().isEmpty()) {
            articleMapper.insertAuthor(article.getAuthors(), article.get_id());
        }
        return article;
    }

    @Override
    public void deleteArticle(Long id) {
        articleMapper.deleteArticle(id);
    }
    @Override
    public void updateArticle(Article article){
        articleMapper.updateArticle(article);
    }
}
