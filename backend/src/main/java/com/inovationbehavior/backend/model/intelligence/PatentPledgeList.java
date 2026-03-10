package com.inovationbehavior.backend.model.intelligence;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PatentPledgeList {
    List<PatentPledge> lists;
    Long count;
}
