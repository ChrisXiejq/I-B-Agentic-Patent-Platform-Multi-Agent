import { defineStore } from 'pinia';

export const surveyStore = defineStore('survey', {
  state: () => ({
    surveyInfo:{
      patentNo:'',
      companyName:'',
      curInvitationCode:"",
      invitationCode:'043398',
      testInvitationCode: "0413test"
    },
    formA:{
      pAq01: [],
      pAq02: "",
      pAq02: "",
      pAq03: [],
      pAq04: [],
      pAq05: [],
      pAq06: [],
      pAq0701: "",
      pAq0702: "",
      pAq08: "",
      pAq09: [],
      pAq10: "",
      pAq11: [],
    },

  }),
})

export const identificationStore = defineStore('form', {
  state: () => ({
    identification_form : {
        p1q1: "",
        p1q2: "",
        p1q3: [],
        p1q4: "",
        p1q5: "",
        p1q6: [],
        p1q7: [],
        p1q8: "",
        p1q9: [],
        p1q10: "",
        p1q11: [],
        p1q12: "",
        p1q13: [],
        p1q14: "",
        p1q15: "",
        p1q16: "",
        p1q17: [],
        p1q18: [],
        p1q19: "",
    }
  }),
  actions: {
    updateForm(fieldName, value) {
      this[fieldName] = value;
    },
    resetForm() {
      Object.keys(this.$state).forEach(key => {
        this[key] = '';
      });
    }
  }
});

export function tableRowChange (tableData, colIndex, currentRow) {  
  // 当前列中其他行取消选中  
  tableData.forEach(otherRow => {  
    if (otherRow !== currentRow) {  
      otherRow.selection[colIndex] = false;  
    }  
  });  
  // 选中列补一个选中状态  
  //currentRow.selection[colIndex] = true;  
  currentRow.selection[colIndex] = !!currentRow.selection[colIndex];  
}

export function tableColChange(tableData, rowIndex, currentCol) {  
  const row = tableData[rowIndex];  
  // 遍历当前行的所有列，取消选中除了目标列以外的所有列  
  row.selection.forEach((isSelected, colIndex) => {  
    if (colIndex !== currentCol) {  
      row.selection[colIndex] = false;  
    }  
  }); 
  //row.selection[currentCol] = true;
  row.selection[currentCol] = !!row.selection[currentCol];
}