package com.inovationbehavior.backend.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.sql.Timestamp;
import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Patent {
    Integer _id;
    String no;
    String name;
    String summary;
    String link;
    String appln_application;
    Timestamp update_time;
    List<String> pdfs;
    String type;
    Integer status;
    Integer exist;
    String agency;
}
