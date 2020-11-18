package NEW_JOURNEY_IN_WORK.array;

public class LC_922 {
    public static int[] sortArrayByParityII(int[] A) {
        for(int i = 0;i<A.length;i++){
            // 奇数位置上为偶数，找到后面的奇数并交换
            if((i%2 == 1) && (A[i]%2 == 0)){
                int j = i + 1;
                while(j < A.length){
                    if(A[j]%2 != 0){
                        // 交换位置
                        int tmp = A[i];
                        A[i] = A[j];
                        A[j] = tmp;
                        break;
                    }
                    j++;
                }
            }
            // 偶数位置上为奇数，找到后面的偶数并交换
            if((i%2 == 0)&&(A[i]%2 != 0)){
                int j = i + 1;
                while(j < A.length){
                    if(A[j]%2 == 0){
                        // 交换位置
                        int tmp = A[i];
                        A[i] = A[j];
                        A[j] = tmp;
                        break;
                    }
                    j++;
                }
            }
        }
        return A;
    }

    public static void main(String[] args) {
        sortArrayByParityII(new int[]{4,2,5,7});
    }
}
