package com.inovationbehavior.backend.service.intf;

import com.inovationbehavior.backend.model.economic.Article;
import com.inovationbehavior.backend.model.economic.ArticleList;
import com.inovationbehavior.backend.model.intelligence.Course;
import com.inovationbehavior.backend.model.intelligence.CourseList;
import com.inovationbehavior.backend.model.intelligence.PatentPledge;
import com.inovationbehavior.backend.model.intelligence.PatentPledgeList;
import org.springframework.stereotype.Service;

@Service
public interface EconomicService {
    ArticleList getArticleByPage(Integer page, Integer size, String type);

    Article getArticleById(Long id);
}
