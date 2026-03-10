package com.inovationbehavior.backend.service.impl;

import com.inovationbehavior.backend.mapper.EconomicMapper;
import com.inovationbehavior.backend.mapper.IntelligenceMapper;
import com.inovationbehavior.backend.model.economic.Article;
import com.inovationbehavior.backend.model.economic.ArticleList;
import com.inovationbehavior.backend.model.economic.Author;
import com.inovationbehavior.backend.model.intelligence.*;
import com.inovationbehavior.backend.service.intf.EconomicService;
import com.inovationbehavior.backend.service.intf.IntelligenceService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@Slf4j
public class EconomicServiceImpl implements EconomicService {
    @Autowired
    private EconomicMapper economicMapper;

    @Override
    public ArticleList getArticleByPage(Integer page, Integer size, String type) {
        Long count = economicMapper.count(type);
        Integer start = (page - 1) * size;
        List<Article> articles = economicMapper.getArticleByPage(start,size,type);
        for (Article article : articles) {
            List<Author> authors = economicMapper.getAuthorById(article.get_id());
            article.setAuthors(authors);
        }
        ArticleList result = new ArticleList();
        result.setCount(count);
        result.setLists(articles);
        return result;
    }

    @Override
    public Article getArticleById(Long id) {
        Article result = economicMapper.getArticleById(id);
        List<Author> authors = economicMapper.getAuthorById(id);
        result.setAuthors(authors);
        return result;
    }
}
