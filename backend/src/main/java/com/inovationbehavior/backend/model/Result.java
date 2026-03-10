package com.inovationbehavior.backend.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Result {
    private Integer code;  // 响应码 1为成功 0为失败
    private String msg;  // 响应信息 "success"或错误信息
    private Object data;  // 返回数据
    public static Result success(){
        return new Result(1,"success",null);
    }  // 无返回数据 成功
    public static Result success(Object data){
        return new Result(1,"success",data);
    }  // 有返回数据 成功
    public static Result error(String msg){
        return new Result(0,msg,null);
    }  // 失败
}
