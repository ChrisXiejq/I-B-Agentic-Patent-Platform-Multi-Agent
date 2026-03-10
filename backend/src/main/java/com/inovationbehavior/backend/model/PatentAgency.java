package com.inovationbehavior.backend.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PatentAgency {
    private Integer id;
    private String appln_id;
    private String agency;
    private String type;
}
