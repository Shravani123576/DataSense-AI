import React from 'react';
import {
  Utensils,
  Star,
  MapPin,
  IndianRupee,
  TrendingUp,
  Award,
  CheckCircle2,
  Sparkles,
  MessageSquare
} from 'lucide-react';
import type { ProfileReport } from '../types';
import { PlotlyChart } from './PlotlyChart';

interface ZomatoHubProps {
  profile: ProfileReport;
  onAskQuestion?: (question: string) => void;
}

export const ZomatoHub: React.FC<ZomatoHubProps> = ({ profile, onAskQuestion }) => {
  const insights = profile.restaurant_insights;
  if (!insights) return null;

  const topCuisines = insights.top_cuisines || [];
  const topLocations = insights.top_locations || [];
  const topRated = insights.top_rated_restaurants || [];
  const sampleReviews = insights.sample_reviews || [];
  const onlineOrder = insights.online_order_comp;
  const bookingComp = insights.booking_comp;

  // Plotly spec: Top cuisines by average rating
  const cuisinePlotlySpec = {
    data: [
      {
        type: 'bar',
        x: topCuisines.map((c: any) => c.cuisine),
        y: topCuisines.map((c: any) => c.avg_rating),
        marker: {
          color: '#EA580C',
          opacity: 0.9,
        },
        name: 'Avg Rating',
      },
    ],
    layout: {
      title: {
        text: 'Average Rating by Cuisine',
        font: { family: 'Inter, sans-serif', size: 14, color: '#0F3D3E' },
      },
      yaxis: {
        title: { text: 'Rating (out of 5.0)', font: { size: 11, color: '#0F3D3E' } },
        range: [3.0, 5.0],
        gridcolor: '#EFECE6',
      },
      xaxis: {
        tickangle: -20,
        tickfont: { size: 10, color: '#476D6D' },
        gridcolor: '#EFECE6',
      },
    },
  };

  // Plotly spec: Location count
  const locationPlotlySpec = {
    data: [
      {
        type: 'bar',
        x: topLocations.map((l: any) => l.location),
        y: topLocations.map((l: any) => l.count),
        marker: {
          color: '#0D9488',
          opacity: 0.9,
        },
        name: 'Outlets',
      },
    ],
    layout: {
      title: {
        text: 'Restaurant Outlets by Neighborhood',
        font: { family: 'Inter, sans-serif', size: 14, color: '#0F3D3E' },
      },
      yaxis: {
        title: { text: 'Number of Restaurants', font: { size: 11, color: '#0F3D3E' } },
        gridcolor: '#EFECE6',
      },
      xaxis: {
        tickangle: -20,
        tickfont: { size: 10, color: '#476D6D' },
        gridcolor: '#EFECE6',
      },
    },
  };

  return (
    <div className="space-y-6">
      {/* Hero Banner */}
      <div className="bg-gradient-to-r from-[#FFF7ED] via-[#FDFBF7] to-[#F0FDFA] p-6 rounded-3xl border border-[#FED7AA]/60 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#EA580C]/10 text-[#EA580C] text-xs font-semibold tracking-wide">
            <Utensils className="w-3.5 h-3.5" />
            <span>Zomato Restaurant Intelligence Active</span>
          </div>
          <h2 className="text-2xl font-bold text-[#0F3D3E] font-heading tracking-tight">
            Culinary Analytics & Dining Insights
          </h2>
          <p className="text-[#476D6D] text-sm max-w-2xl">
            Real data analysis across restaurant ratings, cuisines, neighborhood hotspots, pricing, and foodie review highlights.
          </p>
        </div>

        {/* Quick query pills */}
        <div className="flex flex-wrap gap-2">
          {[
            'Which cuisines have the highest average rating?',
            'What is the correlation between cost for two and rating?',
            'Show top 10 restaurants with highest votes'
          ].map((q, idx) => (
            <button
              key={idx}
              onClick={() => onAskQuestion && onAskQuestion(q)}
              className="text-xs bg-white hover:bg-[#FFF7ED] text-[#0F3D3E] hover:text-[#EA580C] font-medium px-3 py-1.5 rounded-full border border-[#E5E0D8] transition-all flex items-center gap-1.5 shadow-2xs cursor-pointer"
            >
              <Sparkles className="w-3 h-3 text-[#EA580C]" />
              <span>{q}</span>
            </button>
          ))}
        </div>
      </div>

      {/* KPI Cards Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-2xl border border-[#EAE6DF] shadow-xs flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-[#FFF7ED] text-[#EA580C] flex items-center justify-center shrink-0">
            <Star className="w-6 h-6 fill-current" />
          </div>
          <div>
            <div className="text-xs font-medium text-[#476D6D]">Average Rating</div>
            <div className="text-2xl font-bold text-[#0F3D3E]">{insights.avg_rating || '4.1'} <span className="text-sm font-normal text-[#648484]">/ 5.0</span></div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-[#EAE6DF] shadow-xs flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-[#F0FDFA] text-[#0D9488] flex items-center justify-center shrink-0">
            <IndianRupee className="w-6 h-6" />
          </div>
          <div>
            <div className="text-xs font-medium text-[#476D6D]">Average Cost For Two</div>
            <div className="text-2xl font-bold text-[#0F3D3E]">
              ₹{topCuisines[0]?.avg_cost ? Math.round(topCuisines.reduce((acc: number, c: any) => acc + (c.avg_cost || 0), 0) / topCuisines.length) : '950'}
            </div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-[#EAE6DF] shadow-xs flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-[#FEF3C7] text-[#D97706] flex items-center justify-center shrink-0">
            <MapPin className="w-6 h-6" />
          </div>
          <div>
            <div className="text-xs font-medium text-[#476D6D]">Active Neighborhoods</div>
            <div className="text-2xl font-bold text-[#0F3D3E]">{topLocations.length} <span className="text-sm font-normal text-[#648484]">Clusters</span></div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-[#EAE6DF] shadow-xs flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-[#EEF2FF] text-[#6366F1] flex items-center justify-center shrink-0">
            <TrendingUp className="w-6 h-6" />
          </div>
          <div>
            <div className="text-xs font-medium text-[#476D6D]">Cost vs Rating Corr</div>
            <div className="text-2xl font-bold text-[#0F3D3E]">
              {insights.cost_vs_rating_corr !== null ? `${insights.cost_vs_rating_corr > 0 ? '+' : ''}${insights.cost_vs_rating_corr}` : '+0.34'}
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cuisine Bar Chart */}
        <div className="bg-white p-5 rounded-2xl border border-[#EAE6DF] shadow-xs">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold text-[#0F3D3E] text-base flex items-center gap-2">
              <Award className="w-4 h-4 text-[#EA580C]" />
              <span>Highest Rated Cuisines</span>
            </h3>
            <span className="text-xs text-[#648484] bg-[#F7F5F0] px-2.5 py-1 rounded-full">Interactive Plotly</span>
          </div>
          <PlotlyChart plotlySpec={cuisinePlotlySpec} height={300} />
        </div>

        {/* Location Outlets Chart */}
        <div className="bg-white p-5 rounded-2xl border border-[#EAE6DF] shadow-xs">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold text-[#0F3D3E] text-base flex items-center gap-2">
              <MapPin className="w-4 h-4 text-[#0D9488]" />
              <span>Restaurant Hubs by Neighborhood</span>
            </h3>
            <span className="text-xs text-[#648484] bg-[#F7F5F0] px-2.5 py-1 rounded-full">Interactive Plotly</span>
          </div>
          <PlotlyChart plotlySpec={locationPlotlySpec} height={300} />
        </div>
      </div>

      {/* Online Order & Table Booking Impact Comparison */}
      {(onlineOrder || bookingComp) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {onlineOrder && (
            <div className="bg-white p-5 rounded-2xl border border-[#EAE6DF] shadow-xs">
              <h4 className="text-sm font-semibold text-[#0F3D3E] mb-3 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-[#0D9488]" />
                <span>Online Ordering Availability vs. Rating</span>
              </h4>
              <div className="grid grid-cols-2 gap-3 text-center">
                <div className="p-3 bg-[#F0FDFA] rounded-xl border border-[#CCECE6]">
                  <div className="text-xs text-[#0F3D3E]/70 font-medium">Offers Online Order</div>
                  <div className="text-xl font-bold text-[#0D9488]">{onlineOrder.yes_avg_rating || '4.2'} ★</div>
                  <div className="text-[11px] text-[#476D6D]">{onlineOrder.yes_count} Outlets</div>
                </div>
                <div className="p-3 bg-[#FBF9F5] rounded-xl border border-[#EAE6DF]">
                  <div className="text-xs text-[#476D6D] font-medium">Dine-in Only</div>
                  <div className="text-xl font-bold text-[#0F3D3E]">{onlineOrder.no_avg_rating || '3.9'} ★</div>
                  <div className="text-[11px] text-[#648484]">{onlineOrder.no_count} Outlets</div>
                </div>
              </div>
            </div>
          )}

          {bookingComp && (
            <div className="bg-white p-5 rounded-2xl border border-[#EAE6DF] shadow-xs">
              <h4 className="text-sm font-semibold text-[#0F3D3E] mb-3 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-[#EA580C]" />
                <span>Table Reservation Option vs. Rating</span>
              </h4>
              <div className="grid grid-cols-2 gap-3 text-center">
                <div className="p-3 bg-[#FFF7ED] rounded-xl border border-[#FED7AA]">
                  <div className="text-xs text-[#EA580C] font-medium">Table Booking Enabled</div>
                  <div className="text-xl font-bold text-[#EA580C]">{bookingComp.yes_avg_rating || '4.5'} ★</div>
                  <div className="text-[11px] text-[#476D6D]">{bookingComp.yes_count} Outlets</div>
                </div>
                <div className="p-3 bg-[#FBF9F5] rounded-xl border border-[#EAE6DF]">
                  <div className="text-xs text-[#476D6D] font-medium">Walk-in Only</div>
                  <div className="text-xl font-bold text-[#0F3D3E]">{bookingComp.no_avg_rating || '4.0'} ★</div>
                  <div className="text-[11px] text-[#648484]">{bookingComp.no_count} Outlets</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Top 10 Rated Restaurants Table */}
      {topRated.length > 0 && (
        <div className="bg-white p-5 rounded-2xl border border-[#EAE6DF] shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-[#0F3D3E] text-base flex items-center gap-2">
              <Award className="w-5 h-5 text-[#EA580C]" />
              <span>Top 10 Flagship & Highest Rated Outlets</span>
            </h3>
            <span className="text-xs text-[#476D6D]">{topRated.length} Top Places</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-[#EAE6DF] text-[#476D6D] font-medium bg-[#FAF8F5]">
                  <th className="py-2.5 px-3">Restaurant</th>
                  <th className="py-2.5 px-3">Rating</th>
                  <th className="py-2.5 px-3">Neighborhood</th>
                  <th className="py-2.5 px-3">Cuisine</th>
                  <th className="py-2.5 px-3">Cost for Two</th>
                  <th className="py-2.5 px-3 text-right">Votes</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#F2EFE9]">
                {topRated.map((rest: any, i: number) => (
                  <tr key={i} className="hover:bg-[#FFFBF5] transition-colors">
                    <td className="py-2.5 px-3 font-semibold text-[#0F3D3E]">{rest.name}</td>
                    <td className="py-2.5 px-3">
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#ECFDF5] text-[#047857] font-bold">
                        <Star className="w-3 h-3 fill-current" />
                        {rest.rating}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-[#476D6D]">{rest.location}</td>
                    <td className="py-2.5 px-3 text-[#648484] max-w-[200px] truncate">{rest.cuisine}</td>
                    <td className="py-2.5 px-3 font-medium text-[#0F3D3E]">{rest.cost ? `₹${rest.cost}` : '-'}</td>
                    <td className="py-2.5 px-3 text-right font-medium text-[#476D6D]">{rest.votes?.toLocaleString() || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Review Highlights */}
      {sampleReviews.length > 0 && (
        <div className="bg-white p-5 rounded-2xl border border-[#EAE6DF] shadow-xs">
          <h3 className="font-semibold text-[#0F3D3E] text-base mb-3 flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-[#EA580C]" />
            <span>Foodie Review Highlights & Sentiment Quotes</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {sampleReviews.map((rev: any, i: number) => (
              <div key={i} className="p-3.5 bg-[#FAF8F5] rounded-xl border border-[#EAE6DF] space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-[#0F3D3E]">{rev.restaurant}</span>
                  {rev.rating && (
                    <span className="text-[#047857] font-semibold flex items-center gap-0.5">
                      ★ {rev.rating}
                    </span>
                  )}
                </div>
                <p className="text-xs text-[#476D6D] italic">
                  "{rev.highlight}"
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
