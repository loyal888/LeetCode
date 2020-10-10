package NEW_JOURNEY_IN_WORK.array;

public class LC_17_10 {
    public static  int majorityElement(int[] nums) {
        if(nums == null){
            return -1;
        }
        // 投票法
        int count = 0;// 用于记录个数
        int num = 0;// 记录众数
        for(int i = 0;i < nums.length;i++){
            if(count == 0){
                num = nums[i];
                count++;
            }else{
                if(nums[i] == num){
                    count++;
                }else{
                    count--;
                }
            }
        }
        if(count <= 0){
            return -1;
        }
        // 进行验证是不是主要元素
        int main = 0;
        for(int i = 0;i < nums.length;i++){
            if(num == nums[i]){
                main++;
            }
        }
        return main>nums.length/2?num:-1;
    }

    public static void main(String[] args) {
        majorityElement(new int[]{3,2,3});
    }
}

