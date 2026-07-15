//点击确定
   confirmBtn() {
         const params = { //总的提交信息
             id: this.apId, //混合传过来  业务流id
         }
         let that = this;
         faceSignSave(params).then(res => {
             if (res.code == 0) {
                //do something
             } else {
                //do something
             }
         })
     }