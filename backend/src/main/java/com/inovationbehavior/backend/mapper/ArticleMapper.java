package com.inovationbehavior.backend.mapper;

import com.inovationbehavior.backend.model.economic.Article;
import com.inovationbehavior.backend.model.economic.Author;
import com.inovationbehavior.backend.provider.ArticleSqlProvider;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface ArticleMapper {
    @Insert("INSERT INTO article (type, title_en, title_zn, title_de, abstract_en, abstract_zn, abstract_de, url) " +
            "VALUES (#{article.type}, #{article.title_en}, #{article.title_zn}, " +
            "#{article.title_de}, #{article.abstract_en}, #{article.abstract_zn},"+
            "#{article.abstract_de},#{article.url})")
    @Options(useGeneratedKeys = true, keyProperty = "article._id")
    Long insertArticle(@Param("article") Article article);

//    @Update("UPDATE article SET " +
////            "type = #{article.type}, " +
////            "title_en = #{article.title_en}, " +
////            "title_zn = #{article.title_zn}, " +
////            "title_de = #{article.title_de}, " +
////            "abstract_en = #{article.abstract_en}, " +
////            "abstract_zn = #{article.abstract_zn}, " +
////            "abstract_de = #{article.abstract_de}, " +
////            "url = #{article.url} " +
////            "WHERE _id = #{article._id}")
////    int updateArticle(@Param("article") Article article);
    @UpdateProvider(type = ArticleSqlProvider.class, method = "updateArticleSql")
    int updateArticle(@Param("article") Article article);

    @Insert({
            "<script>",
            "INSERT INTO ib.article_author (name_en, name_zn, name_de, article_id) VALUES ",
            "<foreach collection='lecturers' item='lecturer' index='index' separator=','>",
            "(#{lecturer.name_en}, #{lecturer.name_zn}, #{lecturer.name_de}, #{articleID})",
            "</foreach>",
            "</script>"
    })
    int insertAuthor(@Param("lecturers") List<Author> lecturers, @Param("articleID") Long articleID);

    @Delete("""
        DELETE FROM article
        WHERE _id = #{article_id};
    """)
    void deleteArticle(@Param("article_id") Long article_id);
}
