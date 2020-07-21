//package sort
//
//class QuikSort {
//    companion object {
//        fun QuickSort(num: IntArray, left: Int, right: Int) {
//            //如果left等于right，即数组只有一个元素，直接返回
//            if (left >= right) {
//                return
//            }
//
//
//            //设置最左边的元素为基准值
//            val key = num[left]
//            //数组中比key小的放在左边，比key大的放在右边，key值下标为i
//            var i = left
//            var j = right
//            while (i < j) { //j向左移，直到遇到比key小的值
//                while (num[j] >= key && i < j) {
//                    j--
//                }
//                //i向右移，直到遇到比key大的值
//                while (num[i] <= key && i < j) {
//                    i++
//                }
//                //i和j指向的元素交换
//                if (i < j) {
//                    val temp = num[i]
//                    num[i] = num[j]
//                    num[j] = temp
//                }
//            }
//            num[left] = num[i]
//            num[i] = key
//            QuickSort(num, left, i - 1)
//            QuickSort(num, i + 1, right)
//        }
//
//        fun getHalfNums(nums: List<Int>): Int {
//            var length: Int = nums.size / 2
//            var map = mutableMapOf<Int, Int>()
//            for (num in nums) {
//                map[num]?.let {
//                    map[num] = it.plus(1)
//                    map[num]?.let {
//                        if (it > length) {
//                            return num
//                        }
//                    }
//                }
//                map[num] ?: map.put(num, 1)
//            }
//            return -1
//        }
//
//
//    }
//
//    fun main(string: Array<String>) {
//        val halfNums = QuikSort.getHalfNums(listOf(1, 22, 2, 2, 2, 2, 1))
//        print(halfNums)
//    }
//}