package com.inovationbehavior.backend.mapper;

import com.inovationbehavior.backend.model.news.IbNews;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface IbNewsMapper {
    @Insert("INSERT INTO ib_news (title_zn,url_zn,cover,time) " +
            "VALUES (#{ibNews.titleZn},#{ibNews.urlZn},#{ibNews.cover},#{ibNews.time})")
    @Options(useGeneratedKeys = true, keyProperty = "ibNews.id")
    void insertIbNews(@Param("ibNews") IbNews ibNews);

    @Select("SELECT id,title_zn,title_en, url_zn,url_en, cover, time FROM ib_news WHERE id = #{id}")
    @Results({
            @Result(property = "id", column = "id"),
            @Result(property = "titleZn", column = "title_zn"),
            @Result(property = "titleEn", column = "title_en"),
            @Result(property = "urlZn", column = "url_zn"),
            @Result(property = "urlEn", column = "url_en"),
            @Result(property = "cover", column = "cover"),
            @Result(property = "time", column = "time")
    })
    IbNews getNewsById(Integer id);

    @Select("SELECT id,title_zn,title_en, url_zn,url_en, cover , time FROM ib_news")
    @Results({
            @Result(property = "id", column = "id"),
            @Result(property = "titleZn", column = "title_zn"),
            @Result(property = "titleEn", column = "title_en"),
            @Result(property = "urlZn", column = "url_zn"),
            @Result(property = "urlEn", column = "url_en"),
            @Result(property = "cover", column = "cover"),
            @Result(property = "time", column = "time")
    })
    List<IbNews> getNewsList();

    @Delete("""
        DELETE FROM ib_news
        WHERE id = #{id}
""")
    void deleteNewsById(Integer id);
}
