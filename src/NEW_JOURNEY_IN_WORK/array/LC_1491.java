package NEW_JOURNEY_IN_WORK.array;

public class LC_1491 {
    // /**
    // * 先排序，后输出
    // */
    // public double average(int[] salary) {
    //     double sum = 0 ;
    //     // 冒泡排序
    //     for(int i = 0; i < salary.length-1;i++){
    //         for(int j = 0; j < salary.length-1-i;j++){
    //             if(salary[j] > salary[j+1]){
    //                 int tmp = salary[j];
    //                 salary[j] = salary[j+1];
    //                 salary[j+1] = tmp;
    //             }
    //         }
    //     }

    //     // 去掉最大和最小
    //     for(int i = 1; i < salary.length-1;i++){
    //         sum += salary[i];
    //     }
    //     return sum/(salary.length-2);
    // }


    /**
     *
     * 记录最大最小值，然后求值
     * O（n） & O（1）
     */
    public double average(int[] salary) {
        double sum = 0;
        int min=salary[0],max = salary[0];
        for(int i = 0; i < salary.length;i++){
            if(salary[i]>max){
                max = salary[i];
            }
            if(salary[i]<min){
                min = salary[i];
            }
            sum += salary[i];
        }
        return (sum-min-max)/(salary.length-2);

    }
}
