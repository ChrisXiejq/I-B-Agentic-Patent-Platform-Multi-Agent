package com.inovationbehavior.backend.model.survey;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class AwardInfo {
    private String patentNo;
    private String award;
    private String address;
    private String postCode;
    private String receiverName;
    private String phoneNumber;
}
